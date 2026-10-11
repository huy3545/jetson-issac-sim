"""Stage 7D-1 passive six-module contract trace.

This is intentionally a dry run: geometric qualification can be demonstrated
for synthetic boundaries, while strict metric admission and execution remain
blocked.  No controller or legacy CSV logger is imported here.
"""
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import ObservationStatus, VisionResult


SCHEMA_VERSION="1.1"

class DryRunStatus(str,Enum): ACCEPTED="ACCEPTED"; REJECTED="REJECTED"; BLOCKED="BLOCKED"; ERROR="ERROR"; SKIPPED="SKIPPED"

@dataclass(frozen=True)
class DryRunStageRecord:
    schema_version:str; run_id:str; correlation_id:str; stage_id:str; status:DryRunStatus
    source_frame_id:str|int|None; timestamp:float|None; clock_domain:str|None
    coordinate_frame:str|None; units:str|None; config_version:str|None
    source_path_id:str|None; source_segment_ids:tuple[str,...]
    calibration_id:str|None; transform_id:str|None; provenance:tuple[str,...]
    confidence:float|None; uncertainty_status:str|None; geometry_status:str|None
    admission_status:str; reason_codes:tuple[str,...]; execution_authorized:bool=False
    def __post_init__(self):
        if self.schema_version!=SCHEMA_VERSION: raise ValueError("unsupported schema")
        if not self.run_id or not self.correlation_id or not self.stage_id: raise ValueError("missing trace identity")
        if self.execution_authorized: raise ValueError("dry run cannot authorize execution")

@dataclass(frozen=True)
class SixModuleDryRunResult:
    schema_version:str; run_id:str; correlation_id:str; records:tuple[DryRunStageRecord,...]
    final_status:DryRunStatus; audit_complete:bool; execution_authorized:bool=False
    def __post_init__(self):
        if self.schema_version!=SCHEMA_VERSION or len(self.records)!=6: raise ValueError("six records are required")
        if self.execution_authorized or any(record.execution_authorized for record in self.records): raise ValueError("execution authorization forbidden")
        if any(record.run_id!=self.run_id or record.correlation_id!=self.correlation_id for record in self.records): raise ValueError("trace identity mismatch")

class InMemorySixModuleAuditSink:
    def __init__(self): self.records=[]
    def emit(self,record): self.records.append(record)

class SixModuleDryRunHarness:
    """Trace Vision→Path→Motion→Safety→Control simulation→Audit safely."""
    ORDER=("vision","path","motion","safety","machine_control_simulation","data_logging")
    def __init__(self,audit_sink=None):
        self.audit_sink=audit_sink or InMemorySixModuleAuditSink()
        self.path_adapter=VisionPathAdapter(solver=CornerAwareTwoLineSolver(),require_point_provenance=True)

    def run(self,vision,*,run_id=None,correlation_id=None,calibration_id=None,transform_id=None,
            uncertainty_status="UNKNOWN",expected_frame_id=None,expected_timestamp=None,
            expected_clock_domain=None,config_version="stage7d1-synthetic"):
        run_id=run_id or uuid4().hex; correlation_id=correlation_id or run_id
        records=[]
        if not isinstance(vision,VisionResult):
            records.append(self._record(run_id,correlation_id,"vision",DryRunStatus.ERROR,None,("INVALID_VISION_RESULT",),config_version=config_version))
            return self._finish(records,run_id,correlation_id)
        identity=self._identity(vision,calibration_id,transform_id,uncertainty_status,config_version)
        vision_reasons=self._vision_reasons(vision,expected_frame_id,expected_timestamp,expected_clock_domain)
        confidence=min(vision.left.confidence,vision.right.confidence)
        provenance=self._provenance(vision)
        records.append(self._record(run_id,correlation_id,"vision",DryRunStatus.ACCEPTED if not vision_reasons else DryRunStatus.REJECTED,vision,tuple(vision_reasons),confidence=confidence,provenance=provenance,geometry_status="OBSERVED" if not vision_reasons else "UNTRUSTED",**identity))

        if vision_reasons:
            records.append(self._record(run_id,correlation_id,"path",DryRunStatus.SKIPPED,vision,("VISION_DEPENDENCY_REJECTED",),confidence=confidence,provenance=provenance,geometry_status="UNAVAILABLE",**identity))
        else:
            path=self.path_adapter.plan(vision)
            if path.status is VisionPathStatus.VERIFIED:
                segment_ids=("synthetic-xy-segment-0",)
                records.append(self._record(run_id,correlation_id,"path",DryRunStatus.BLOCKED,vision,("STRICT_GATE_B_BLOCKED","CALIBRATION_UNVERIFIED","XYZ_UNCERTAINTY_UNKNOWN"),confidence=confidence,provenance=provenance,geometry_status="SYNTHETIC_GEOMETRY_VERIFIED",source_path_id="synthetic-vision-path",source_segment_ids=segment_ids,**identity))
            else:
                records.append(self._record(run_id,correlation_id,"path",DryRunStatus.REJECTED,vision,path.reason_codes,confidence=confidence,provenance=provenance,geometry_status="REJECTED",**identity))
        path_record=records[-1]
        records.append(self._record(run_id,correlation_id,"motion",DryRunStatus.BLOCKED,vision,("UPSTREAM_STRICT_ADMISSION_BLOCKED",),confidence=confidence,provenance=provenance,geometry_status=path_record.geometry_status,source_path_id=path_record.source_path_id,source_segment_ids=path_record.source_segment_ids,**identity))
        records.append(self._record(run_id,correlation_id,"safety",DryRunStatus.BLOCKED,vision,("UPSTREAM_BLOCKED","EXECUTION_NOT_AUTHORIZED"),confidence=confidence,provenance=provenance,geometry_status=path_record.geometry_status,source_path_id=path_record.source_path_id,source_segment_ids=path_record.source_segment_ids,**identity))
        records.append(self._record(run_id,correlation_id,"machine_control_simulation",DryRunStatus.BLOCKED,vision,("EXECUTION_BLOCKED","PASSIVE_SIMULATION_ONLY"),confidence=confidence,provenance=provenance,geometry_status=path_record.geometry_status,source_path_id=path_record.source_path_id,source_segment_ids=path_record.source_segment_ids,**identity))
        return self._finish(records,run_id,correlation_id,vision=vision,identity=identity,confidence=confidence,provenance=provenance,path_record=path_record)

    def _finish(self,records,run_id,correlation_id,*,vision=None,identity=None,confidence=None,provenance=(),path_record=None):
        audit_complete=True
        if vision is None:
            identity={"calibration_id":None,"transform_id":None,"uncertainty_status":"UNKNOWN","config_version":"stage7d1-synthetic"}; confidence=None
        logging=self._record(run_id,correlation_id,"data_logging",DryRunStatus.ACCEPTED,vision,(),confidence=confidence,provenance=provenance,geometry_status=getattr(path_record,"geometry_status",None),source_path_id=getattr(path_record,"source_path_id",None),source_segment_ids=getattr(path_record,"source_segment_ids",()),**identity)
        try:
            for record in records: self.audit_sink.emit(record)
            self.audit_sink.emit(logging)
        except Exception:
            audit_complete=False
            logging=self._record(run_id,correlation_id,"data_logging",DryRunStatus.ERROR,vision,("AUDIT_LOGGING_FAILED",),confidence=confidence,provenance=provenance,geometry_status=getattr(path_record,"geometry_status",None),source_path_id=getattr(path_record,"source_path_id",None),source_segment_ids=getattr(path_record,"source_segment_ids",()),**identity)
        records.append(logging)
        final=DryRunStatus.ERROR if not audit_complete else (DryRunStatus.BLOCKED if any(item.status is DryRunStatus.BLOCKED for item in records) else DryRunStatus.REJECTED)
        return SixModuleDryRunResult(SCHEMA_VERSION,run_id,correlation_id,tuple(records),final,audit_complete)

    @staticmethod
    def _provenance(vision):
        values=[]
        for name,line in (("left",vision.left),("right",vision.right)):
            if line.point_provenance is None: values.append(f"{name}:POINT_PROVENANCE_UNAVAILABLE")
            else: values.extend(f"{name}:{point.source_index}" if point.source_index is not None else f"{name}:SOURCE_INDEX_UNAVAILABLE" for point in line.point_provenance)
        return tuple(values)
    @staticmethod
    def _identity(vision,calibration_id,transform_id,uncertainty_status,config_version):
        return {"calibration_id":calibration_id,"transform_id":transform_id,"uncertainty_status":uncertainty_status,"config_version":config_version}
    @staticmethod
    def _vision_reasons(vision,expected_frame_id,expected_timestamp,expected_clock_domain):
        reasons=[]
        for line,name in ((vision.left,"LEFT"),(vision.right,"RIGHT")):
            if line.observation_status is not ObservationStatus.OBSERVED: reasons.append(f"{name}_NOT_OBSERVED")
            if not line.verified: reasons.append(f"{name}_NOT_VERIFIED")
            if line.missing_ranges: reasons.append(f"{name}_GAPPED")
            if line.point_provenance is None or not all(point.lineage_available for point in line.point_provenance): reasons.append(f"{name}_POINT_PROVENANCE_UNAVAILABLE")
        if not vision.time_sync_valid: reasons.append("TIME_SYNC_INVALID")
        if expected_frame_id is not None and vision.frame_id!=expected_frame_id: reasons.append("SOURCE_FRAME_ID_MISMATCH")
        if expected_timestamp is not None and vision.frame_timestamp!=expected_timestamp: reasons.append("TIMESTAMP_MISMATCH")
        if expected_clock_domain is not None and vision.frame_clock_domain!=expected_clock_domain: reasons.append("CLOCK_DOMAIN_MISMATCH")
        return tuple(dict.fromkeys(reasons))
    @staticmethod
    def _record(run,correlation,stage,status,vision,reasons,*,confidence=None,provenance=(),geometry_status=None,source_path_id=None,source_segment_ids=(),calibration_id=None,transform_id=None,uncertainty_status="UNKNOWN",config_version=None):
        return DryRunStageRecord(SCHEMA_VERSION,run,correlation,stage,status,getattr(vision,"frame_id",None),getattr(vision,"frame_timestamp",None),getattr(vision,"frame_clock_domain",None),getattr(vision,"coordinate_frame",None),getattr(vision,"units",None),config_version,source_path_id,tuple(source_segment_ids),calibration_id,transform_id,tuple(provenance),confidence,uncertainty_status,geometry_status,status.value,tuple(reasons),False)
