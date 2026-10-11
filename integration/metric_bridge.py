"""Evidence-preserving Stage 7C metric bridge; never a strict-admission bypass."""
from dataclasses import dataclass
import numpy as np

from integration.contracts import AdmissionStatus, INTEGRATION_SCHEMA_VERSION
from path_planning.experimental_metric_projection import (MetricGeometryResult, MetricGeometryStatus,
    MetricPointProvenance, MetricUncertainty, MetricUncertaintyStatus)
from vision.contracts import VisionResult

def _readonly(value):
    array=np.ascontiguousarray(np.array(value,dtype=float,copy=True)); return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)

@dataclass(frozen=True)
class MetricBridgeSegment:
    name:str; xyz:np.ndarray|None; source_indices:tuple[int,int]|None; geometry_status:str; uncertainty_status:str; observation_status:str; source_geometry_reference:str|None
    point_provenance:tuple[MetricPointProvenance,...]|None=None; uncertainty:MetricUncertainty=MetricUncertainty(); reason_codes:tuple[str,...]=()
    def __post_init__(self):
        if self.xyz is not None: object.__setattr__(self,"xyz",_readonly(self.xyz))
        if self.point_provenance is not None: object.__setattr__(self,"point_provenance",tuple(self.point_provenance))

@dataclass(frozen=True)
class MetricBridgeResult:
    schema_version:str; run_id:str; correlation_id:str; source_frame_id:str|int|None; timestamp:float|None; clock_domain:str|None; coordinate_frame:str|None; units:str|None; calibration_id:str|None; transform_id:str|None; segments:tuple[MetricBridgeSegment,...]; has_gap:bool; admission_status:AdmissionStatus; reason_codes:tuple[str,...]; execution_authorized:bool=False

class MetricGeometryBridge:
    """Maps only published fields and deliberately always remains BLOCKED."""
    def convert(self,vision,metric,*,run_id,correlation_id):
        if not isinstance(vision,VisionResult): return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,None,None,None,None,None,None,None,(),True,AdmissionStatus.REJECTED,("INVALID_VISION_RESULT",))
        if not isinstance(metric,MetricGeometryResult): return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,vision.frame_id,vision.frame_timestamp,vision.frame_clock_domain,None,None,None,None,(),True,AdmissionStatus.REJECTED,("INVALID_METRIC_GEOMETRY_RESULT",))
        evidence=metric.transform_evidence; calibration_id=getattr(evidence,"intrinsics_id",None); transform_id=getattr(evidence,"extrinsics_id",None); reasons=[]
        if metric.units!="m": reasons.append("UNITS_NOT_METRIC")
        if not metric.reference_frame: reasons.append("REFERENCE_FRAME_MISSING")
        if calibration_id is None: reasons.append("CALIBRATION_ID_MISSING")
        if transform_id is None: reasons.append("TRANSFORM_ID_MISSING")
        if metric.source_id!=str(vision.frame_id): reasons.append("SOURCE_ID_MISMATCH")
        if metric.timestamp!=vision.frame_timestamp or metric.clock_domain!=vision.frame_clock_domain: reasons.append("TIMESTAMP_DOMAIN_MISMATCH")
        if metric.status is not MetricGeometryStatus.UNCERTAIN: reasons.append("METRIC_STATUS_UNEXPECTED")
        if metric.uncertainty_status is not MetricUncertaintyStatus.UNKNOWN: reasons.append("METRIC_UNCERTAINTY_UNEXPECTED")
        segments=[]
        for source in metric.segments:
            line=vision.left if source.name=="left" else vision.right
            lineage=source.point_provenance
            if lineage is None or not all(point.provenance_available for point in lineage): reasons.append("POINT_PROVENANCE_UNAVAILABLE")
            observation=line.observation_status.value  # preserve OCCLUDED and INVALID exactly
            if observation!="OBSERVED": reasons.append("NON_OBSERVED_SOURCE")
            segments.append(MetricBridgeSegment(source.name,source.xyz,source.source_indices,source.status.value,source.uncertainty_status.value,observation,source.source_id,lineage,source.uncertainty,tuple(source.diagnostics)))
        reasons.extend(metric.diagnostics)
        return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,vision.frame_id,metric.timestamp,metric.clock_domain,metric.reference_frame,metric.units,calibration_id,transform_id,tuple(segments),len(segments)!=2,AdmissionStatus.BLOCKED,tuple(dict.fromkeys(reasons)))

class MetricBridgeVerifier:
    """Independent structural comparison; it calls neither projection nor bridge conversion."""
    def verify(self,bridge):
        reasons=[]
        if not isinstance(bridge,MetricBridgeResult) or bridge.schema_version!=INTEGRATION_SCHEMA_VERSION: return ("UNKNOWN_SCHEMA",)
        if bridge.units!="m": reasons.append("UNITS_NOT_METRIC")
        if not bridge.coordinate_frame: reasons.append("REFERENCE_FRAME_MISSING")
        if not bridge.calibration_id: reasons.append("CALIBRATION_ID_MISSING")
        if not bridge.transform_id: reasons.append("TRANSFORM_ID_MISSING")
        if bridge.admission_status is AdmissionStatus.ACCEPTED: reasons.append("TRUST_UPGRADE_FORBIDDEN")
        if any(segment.point_provenance is None or not all(point.provenance_available for point in segment.point_provenance)
               for segment in bridge.segments) and "POINT_PROVENANCE_UNAVAILABLE" not in bridge.reason_codes:
            reasons.append("POINT_PROVENANCE_DISCLOSURE_MISSING")
        return tuple(dict.fromkeys(reasons))
    def verify_lineage(self,vision,metric,bridge):
        reasons=list(self.verify(bridge))
        if not isinstance(vision,VisionResult) or not isinstance(metric,MetricGeometryResult): return tuple(dict.fromkeys(reasons+["INVALID_SOURCE_CONTRACT"]))
        if bridge.source_frame_id!=vision.frame_id or metric.source_id!=str(vision.frame_id): reasons.append("SOURCE_ID_MISMATCH")
        if bridge.timestamp!=vision.frame_timestamp or bridge.clock_domain!=vision.frame_clock_domain: reasons.append("TIMESTAMP_DOMAIN_MISMATCH")
        if len(metric.segments)!=len(bridge.segments): reasons.append("SEGMENT_COUNT_MISMATCH")
        for source,output in zip(metric.segments,bridge.segments):
            line=vision.left if source.name=="left" else vision.right
            if source.name!=output.name: reasons.append("SEGMENT_ORDER_MISMATCH")
            if output.observation_status!=line.observation_status.value: reasons.append("OBSERVATION_STATUS_MISMATCH")
            if source.xyz is None or output.xyz is None or not np.array_equal(source.xyz,output.xyz): reasons.append("XYZ_MISMATCH")
            if source.point_provenance is None or output.point_provenance is None:
                reasons.append("POINT_PROVENANCE_UNAVAILABLE"); continue
            if not all(point.provenance_available for point in source.point_provenance): reasons.append("POINT_PROVENANCE_UNAVAILABLE")
            if len(source.point_provenance)!=len(output.point_provenance): reasons.append("POINT_COUNT_MISMATCH")
            for point,copied in zip(source.point_provenance,output.point_provenance):
                if point!=copied: reasons.append("POINT_LINEAGE_MISMATCH")
                if tuple(point.pixel_xy)!=tuple(copied.pixel_xy): reasons.append("PIXEL_COORDINATE_MISMATCH")
                if point.vision_point is not None and tuple(point.pixel_xy)!=tuple(point.vision_point.pixel_xy): reasons.append("PIXEL_COORDINATE_MISMATCH")
                if point.depth_samples is None: reasons.append("DEPTH_LINEAGE_UNAVAILABLE")
                if point.intrinsics_id!=bridge.calibration_id or point.transform_id!=bridge.transform_id: reasons.append("CALIBRATION_TRANSFORM_MISMATCH")
        return tuple(dict.fromkeys(reasons))
