"""Stage 7B-2 simulation-only plans and independent safety vetoes.

No controller, hardware backend, or Legacy SafetyMonitor is imported here.
"""
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import numpy as np

from .contracts import (INTEGRATION_SCHEMA_VERSION, AdmissionStatus, MotionTimingRecord,
                        PointProvenance, TimingStatus, UncertaintyStatus)
from .validation import ProvenanceResolver

class SafetyStatus(str, Enum):
    ACCEPTED="ACCEPTED"; BLOCKED="BLOCKED"; REJECTED="REJECTED"; ERROR="ERROR"

def _ro_xyz(value,name):
    arr=np.array(value,dtype=float,copy=True)
    if arr.ndim!=2 or arr.shape[1:]!=(3,) or len(arr)==0 or not np.isfinite(arr).all():raise ValueError(f"{name} must be finite Nx3")
    arr.setflags(write=False);return arr
def _ids(values,name):
    if not isinstance(values,tuple) or not values or not all(isinstance(x,str) and x for x in values):raise ValueError(f"{name} must be non-empty tuple[str]")
def _enum(value,typ,name):
    if not isinstance(value,typ):raise ValueError(f"unknown {name}")
def _canonical(value):
    if isinstance(value,Enum):return value.value
    if isinstance(value,np.ndarray):return {"shape":value.shape,"dtype":"float64","bytes":value.astype("<f8",copy=False).tobytes().hex()}
    if isinstance(value,tuple):return [_canonical(item) for item in value]
    if hasattr(value,"__dict__"):return {key:_canonical(item) for key,item in value.__dict__.items()}
    return value
def _hash(payload):return hashlib.sha256(json.dumps(_canonical(payload),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class PathPlan:
    run_id:str; correlation_id:str; path_id:str; source_frame_id:str; timestamp:float; clock_domain:str; coordinate_frame:str; units:str
    calibration_id:str|None; transform_id:str|None; config_version:str; segment_ids:tuple[str,...]; point_ids:tuple[str,...]; point_provenance:tuple[PointProvenance,...]
    coordinates:np.ndarray; geometry_status:str; qualification_status:str; uncertainty_status:UncertaintyStatus; has_gap:bool; schema_version:str=INTEGRATION_SCHEMA_VERSION; content_hash:str=field(init=False)
    def __post_init__(self):
        for name in ("run_id","correlation_id","path_id","source_frame_id","clock_domain","coordinate_frame","units","config_version","geometry_status","qualification_status"):
            if not isinstance(getattr(self,name),str) or not getattr(self,name):raise ValueError(f"missing {name}")
        if not isinstance(self.timestamp,(int,float)) or isinstance(self.timestamp,bool) or not math.isfinite(self.timestamp):raise ValueError("invalid timestamp")
        if self.schema_version!=INTEGRATION_SCHEMA_VERSION:raise ValueError("unsupported schema_version")
        _enum(self.uncertainty_status,UncertaintyStatus,"uncertainty_status");_ids(self.segment_ids,"segment_ids");_ids(self.point_ids,"point_ids")
        arr=_ro_xyz(self.coordinates,"coordinates")
        if len(self.point_ids)!=len(arr) or len(self.point_provenance)!=len(arr):raise ValueError("point metadata count mismatch")
        if tuple(point.point_id for point in self.point_provenance)!=self.point_ids:raise ValueError("point provenance identity mismatch")
        object.__setattr__(self,"coordinates",arr)
        object.__setattr__(self,"content_hash",_hash((self.run_id,self.correlation_id,self.path_id,self.source_frame_id,self.timestamp,self.clock_domain,self.coordinate_frame,self.units,self.calibration_id,self.transform_id,self.config_version,self.segment_ids,self.point_ids,self.point_provenance,arr,self.geometry_status,self.qualification_status,self.uncertainty_status,self.has_gap,self.schema_version)))

@dataclass(frozen=True)
class MotionPlan:
    run_id:str; correlation_id:str; motion_plan_id:str; source_path_id:str; source_path_hash:str; segment_ids:tuple[str,...]; target_ids:tuple[str,...]; target_point_ids:tuple[str,...]
    target_coordinates:np.ndarray; coordinate_frame:str; units:str; timestamp:float; clock_domain:str; config_version:str; timing_records:tuple[MotionTimingRecord,...]; provenance_references:tuple[str,...]; motion_status:str; schema_version:str=INTEGRATION_SCHEMA_VERSION; content_hash:str=field(init=False)
    def __post_init__(self):
        for name in ("run_id","correlation_id","motion_plan_id","source_path_id","source_path_hash","coordinate_frame","units","clock_domain","config_version","motion_status"):
            if not isinstance(getattr(self,name),str) or not getattr(self,name):raise ValueError(f"missing {name}")
        if not isinstance(self.timestamp,(int,float)) or isinstance(self.timestamp,bool) or not math.isfinite(self.timestamp):raise ValueError("invalid timestamp")
        if self.schema_version!=INTEGRATION_SCHEMA_VERSION:raise ValueError("unsupported schema_version")
        _ids(self.segment_ids,"segment_ids");_ids(self.target_ids,"target_ids");_ids(self.target_point_ids,"target_point_ids")
        arr=_ro_xyz(self.target_coordinates,"target_coordinates")
        if len(arr)!=len(self.target_ids) or len(arr)!=len(self.target_point_ids) or len(arr)!=len(self.timing_records):raise ValueError("target metadata count mismatch")
        if not all(isinstance(item,MotionTimingRecord) for item in self.timing_records):raise ValueError("invalid timing_records")
        object.__setattr__(self,"target_coordinates",arr)
        object.__setattr__(self,"content_hash",_hash((self.run_id,self.correlation_id,self.motion_plan_id,self.source_path_id,self.source_path_hash,self.segment_ids,self.target_ids,self.target_point_ids,arr,self.coordinate_frame,self.units,self.timestamp,self.clock_domain,self.config_version,self.timing_records,self.provenance_references,self.motion_status,self.schema_version)))

@dataclass(frozen=True)
class SyntheticMachineLimits:
    config_version:str; workspace_min:tuple[float,float,float]; workspace_max:tuple[float,float,float]; max_step_m:float; max_speed_m_s:float; units:str="m"; real_accuracy_claim:str="NONE"
    def __post_init__(self):
        if not self.config_version or self.units!="m" or self.real_accuracy_claim!="NONE":raise ValueError("invalid synthetic limits identity")
        values=(*self.workspace_min,*self.workspace_max,self.max_step_m,self.max_speed_m_s)
        if len(self.workspace_min)!=3 or len(self.workspace_max)!=3 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values):raise ValueError("invalid machine limits")
        if any(lo>=hi for lo,hi in zip(self.workspace_min,self.workspace_max)) or self.max_step_m<=0 or self.max_speed_m_s<=0:raise ValueError("invalid machine limits")

@dataclass(frozen=True)
class SafetyDecision:
    run_id:str;correlation_id:str;path_id:str;path_hash:str;motion_plan_id:str;motion_plan_hash:str;status:SafetyStatus;reason_codes:tuple[str,...];checked_boundaries:tuple[str,...];safety_config_version:str;timestamp:float;clock_domain:str;schema_version:str=INTEGRATION_SCHEMA_VERSION;execution_authorized:bool=False
    def __post_init__(self):
        _enum(self.status,SafetyStatus,"safety status")
        if self.schema_version!=INTEGRATION_SCHEMA_VERSION or self.execution_authorized:raise ValueError("invalid safety decision")
        if not all(isinstance(x,str) and x for x in self.reason_codes+self.checked_boundaries):raise ValueError("invalid safety decision nested data")

@dataclass(frozen=True)
class SafetyAuditEvent:
    run_id:str;correlation_id:str;stage_id:str;path_hash:str;motion_hash:str;status:SafetyStatus;reason_codes:tuple[str,...];config_version:str;timestamp:float;clock_domain:str

@dataclass(frozen=True)
class BoundaryVeto:
    boundary_id:str;input_identity:str;status:SafetyStatus;reason_codes:tuple[str,...];safety_decision_hash:str|None=None

class IndependentSafetyValidator:
    """Independent evidence and simulation-limit validation; never authorizes motion."""
    def assess(self,path,motion,limits,machine_state,audit_sink=None):
        reasons=[];checked=("schema","identity","provenance","geometry","motion_limits","timing")
        try:
            if not isinstance(path,PathPlan) or not isinstance(motion,MotionPlan) or not isinstance(limits,SyntheticMachineLimits):raise ValueError("INVALID_CONTRACT")
            if machine_state is None:reasons.append("MACHINE_STATE_MISSING")
            if path.run_id!=motion.run_id or path.correlation_id!=motion.correlation_id:reasons.append("RUN_CORRELATION_MISMATCH")
            if motion.source_path_id!=path.path_id or motion.source_path_hash!=path.content_hash:reasons.append("PATH_HASH_MISMATCH")
            if path.coordinate_frame!=motion.coordinate_frame or path.units!=motion.units or path.clock_domain!=motion.clock_domain:reasons.append("FRAME_CLOCK_UNITS_MISMATCH")
            if not path.calibration_id or not path.transform_id:reasons.append("CALIBRATION_TRANSFORM_MISSING")
            if path.has_gap:reasons.append("PATH_GAP")
            if path.qualification_status!="VERIFIED":reasons.append("QUALIFICATION_UNVERIFIED")
            if path.uncertainty_status is not UncertaintyStatus.KNOWN:reasons.append("UNCERTAINTY_UNKNOWN")
            resolver=ProvenanceResolver({item.point_id:item for item in path.point_provenance})
            for point in path.point_ids:reasons.extend(resolver.validate(point).reason_codes)
            if tuple(motion.target_point_ids)!=path.point_ids:reasons.append("TARGET_PATH_REFERENCE_MISMATCH")
            if tuple(motion.segment_ids)!=tuple(path.segment_ids):reasons.append("SEGMENT_ORDER_MISMATCH")
            if any(item.timing_status is TimingStatus.UNSPECIFIED for item in motion.timing_records):reasons.append("TIMING_UNSPECIFIED")
            xyz=motion.target_coordinates
            if ((xyz<limits.workspace_min)|(xyz>limits.workspace_max)).any():reasons.append("WORKSPACE_VIOLATION")
            if len(xyz)>1 and (np.linalg.norm(np.diff(xyz,axis=0),axis=1)>limits.max_step_m).any():reasons.append("TARGET_STEP_EXCEEDED")
            if any(item.speed_m_s is not None and item.speed_m_s>limits.max_speed_m_s for item in motion.timing_records):reasons.append("SPEED_LIMIT_EXCEEDED")
            reasons=tuple(dict.fromkeys(reasons));status=SafetyStatus.ACCEPTED if not reasons else (SafetyStatus.BLOCKED if any(code in reasons for code in ("MACHINE_STATE_MISSING","CALIBRATION_TRANSFORM_MISSING","TIMING_UNSPECIFIED","PATH_GAP","UNCERTAINTY_UNKNOWN")) else SafetyStatus.REJECTED)
        except Exception as exc:
            reasons=("SAFETY_VALIDATOR_ERROR",type(exc).__name__);status=SafetyStatus.ERROR
            path=path if isinstance(path,PathPlan) else None;motion=motion if isinstance(motion,MotionPlan) else None
        decision=SafetyDecision(path.run_id if path else "unknown",path.correlation_id if path else "unknown",path.path_id if path else "unknown",path.content_hash if path else "unknown",motion.motion_plan_id if motion else "unknown",motion.content_hash if motion else "unknown",status,reasons,checked,limits.config_version if isinstance(limits,SyntheticMachineLimits) else "unknown",path.timestamp if path else 0.,path.clock_domain if path else "unknown")
        event=SafetyAuditEvent(decision.run_id,decision.correlation_id,"independent_safety",decision.path_hash,decision.motion_plan_hash,decision.status,decision.reason_codes,decision.safety_config_version,decision.timestamp,decision.clock_domain)
        audit_complete=True
        if audit_sink is not None:
            try:audit_sink.emit(event)
            except Exception:
                audit_complete=False
                decision=SafetyDecision(decision.run_id,decision.correlation_id,decision.path_id,decision.path_hash,decision.motion_plan_id,decision.motion_plan_hash,SafetyStatus.BLOCKED,tuple(dict.fromkeys(decision.reason_codes+("EVIDENCE_INCOMPLETE",))),checked,decision.safety_config_version,decision.timestamp,decision.clock_domain)
        return decision,audit_complete

def apply_boundary_veto(boundary_id,input_identity,upstream_status,safety_decision=None):
    if upstream_status is not SafetyStatus.ACCEPTED:return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("UPSTREAM_NOT_ACCEPTED",))
    if safety_decision is not None and safety_decision.status is not SafetyStatus.ACCEPTED:return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("SAFETY_VETO",),safety_decision.motion_plan_hash)
    return BoundaryVeto(boundary_id,input_identity,SafetyStatus.ACCEPTED,())
