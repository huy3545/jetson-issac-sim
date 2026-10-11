"""Public Stage 7C-1 bridge from MetricGeometryResult without trust upgrades."""
from dataclasses import dataclass
import numpy as np
from integration.contracts import AdmissionStatus, INTEGRATION_SCHEMA_VERSION
from path_planning.experimental_metric_projection import MetricGeometryResult, MetricGeometryStatus, MetricUncertaintyStatus
from vision.contracts import ObservationStatus, VisionResult

def _ro(x):
 a=np.array(x,float,copy=True);a.setflags(write=False);return a
@dataclass(frozen=True)
class MetricBridgeSegment:
 name:str;xyz:np.ndarray|None;source_indices:tuple[int,int]|None;geometry_status:str;uncertainty_status:str;observation_status:str;source_geometry_reference:str|None;reason_codes:tuple[str,...]=()
 def __post_init__(self):
  if self.xyz is not None:object.__setattr__(self,"xyz",_ro(self.xyz))
@dataclass(frozen=True)
class MetricBridgeResult:
 schema_version:str;run_id:str;correlation_id:str;source_frame_id:str|int|None;timestamp:float|None;clock_domain:str|None;coordinate_frame:str|None;units:str|None;calibration_id:str|None;transform_id:str|None;segments:tuple[MetricBridgeSegment,...];has_gap:bool;admission_status:AdmissionStatus;reason_codes:tuple[str,...];execution_authorized:bool=False

class MetricGeometryBridge:
 """Maps only fields published by VisionResult and MetricGeometryResult."""
 def convert(self,vision,metric,*,run_id,correlation_id):
  reasons=[]
  if not isinstance(vision,VisionResult):return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,None,None,None,None,None,None,None,(),True,AdmissionStatus.REJECTED,("INVALID_VISION_RESULT",))
  if not isinstance(metric,MetricGeometryResult):return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,vision.frame_id,vision.frame_timestamp,vision.frame_clock_domain,None,None,None,None,(),True,AdmissionStatus.REJECTED,("INVALID_METRIC_GEOMETRY_RESULT",))
  evidence=metric.transform_evidence
  calibration_id=getattr(evidence,"intrinsics_id",None);transform_id=getattr(evidence,"extrinsics_id",None)
  if metric.units!="m":reasons.append("UNITS_NOT_METRIC")
  if not metric.reference_frame:reasons.append("REFERENCE_FRAME_MISSING")
  if calibration_id is None:reasons.append("CALIBRATION_ID_MISSING")
  if transform_id is None:reasons.append("TRANSFORM_ID_MISSING")
  if metric.source_id!=str(vision.frame_id):reasons.append("SOURCE_ID_MISMATCH")
  if metric.timestamp!=vision.frame_timestamp or metric.clock_domain!=vision.frame_clock_domain:reasons.append("TIMESTAMP_DOMAIN_MISMATCH")
  if metric.status is not MetricGeometryStatus.UNCERTAIN:reasons.append("METRIC_STATUS_UNEXPECTED")
  if metric.uncertainty_status is not MetricUncertaintyStatus.UNKNOWN:reasons.append("METRIC_UNCERTAINTY_UNEXPECTED")
  segs=[]
  for s in metric.segments:
   line=vision.left if s.name=="left" else vision.right
   obs="OBSERVED" if line.observation_status is ObservationStatus.OBSERVED else "PREDICTED"
   segs.append(MetricBridgeSegment(s.name,s.xyz,s.source_indices,s.status.value,s.uncertainty_status.value,obs,s.source_id,tuple(s.diagnostics)))
  # Current public metric result cannot carry every source point lineage.
  reasons.extend(metric.diagnostics);reasons.append("POINT_PROVENANCE_UNAVAILABLE")
  if any(s.observation_status!="OBSERVED" for s in segs):reasons.append("PREDICTED_SOURCE")
  return MetricBridgeResult(INTEGRATION_SCHEMA_VERSION,run_id,correlation_id,vision.frame_id,metric.timestamp,metric.clock_domain,metric.reference_frame,metric.units,calibration_id,transform_id,tuple(segs),len(segs)!=2,AdmissionStatus.BLOCKED,tuple(dict.fromkeys(reasons)))

class MetricBridgeVerifier:
 def verify(self,bridge):
  reasons=[]
  if not isinstance(bridge,MetricBridgeResult) or bridge.schema_version!=INTEGRATION_SCHEMA_VERSION: return ("UNKNOWN_SCHEMA",)
  if bridge.units!="m":reasons.append("UNITS_NOT_METRIC")
  if not bridge.coordinate_frame:reasons.append("REFERENCE_FRAME_MISSING")
  if not bridge.calibration_id:reasons.append("CALIBRATION_ID_MISSING")
  if not bridge.transform_id:reasons.append("TRANSFORM_ID_MISSING")
  if bridge.admission_status is AdmissionStatus.ACCEPTED:reasons.append("TRUST_UPGRADE_FORBIDDEN")
  if "POINT_PROVENANCE_UNAVAILABLE" not in bridge.reason_codes:reasons.append("POINT_PROVENANCE_DISCLOSURE_MISSING")
  return tuple(dict.fromkeys(reasons))
