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
class MachineStateEvidenceStatus(str, Enum):
    KNOWN="KNOWN"; UNKNOWN="UNKNOWN"

def _ro_xyz(value,name):
    arr=np.array(value,dtype=float,copy=True)
    if arr.ndim!=2 or arr.shape[1:]!=(3,) or len(arr)==0 or not np.isfinite(arr).all():raise ValueError(f"{name} must be finite Nx3")
    raw=arr.astype("<f8",copy=False).tobytes(order="C")
    return np.frombuffer(raw,dtype="<f8").reshape(arr.shape)
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
    config_version:str; workspace_min:tuple[float,float,float]; workspace_max:tuple[float,float,float]; max_step_m:float; max_speed_m_s:float; units:str="m"; real_accuracy_claim:str="NONE"; max_path_motion_deviation_m:float|None=None; max_c_step_rad:float|None=None
    def __post_init__(self):
        if not self.config_version or self.units!="m" or self.real_accuracy_claim!="NONE":raise ValueError("invalid synthetic limits identity")
        values=(*self.workspace_min,*self.workspace_max,self.max_step_m,self.max_speed_m_s)
        if len(self.workspace_min)!=3 or len(self.workspace_max)!=3 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values):raise ValueError("invalid machine limits")
        if any(lo>=hi for lo,hi in zip(self.workspace_min,self.workspace_max)) or self.max_step_m<=0 or self.max_speed_m_s<=0:raise ValueError("invalid machine limits")
        for value in (self.max_path_motion_deviation_m,self.max_c_step_rad):
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):raise ValueError("invalid optional machine limit")
    @property
    def limits_hash(self):return _hash((self.config_version,self.workspace_min,self.workspace_max,self.max_step_m,self.max_speed_m_s,self.units,self.real_accuracy_claim,self.max_path_motion_deviation_m,self.max_c_step_rad))

@dataclass(frozen=True)
class SyntheticMachineState:
    run_id:str;correlation_id:str;source_frame_id:str;timestamp:float;clock_domain:str;configuration_id:str;values:tuple[tuple[str,float],...];evidence_status:MachineStateEvidenceStatus;schema_version:str=INTEGRATION_SCHEMA_VERSION;real_accuracy_claim:str="NONE"
    def __post_init__(self):
        if self.schema_version!=INTEGRATION_SCHEMA_VERSION or self.real_accuracy_claim!="NONE":raise ValueError("invalid machine state schema")
        if not isinstance(self.evidence_status,MachineStateEvidenceStatus):raise ValueError("unknown machine state evidence status")
        if not all(isinstance(x,str) and x for x in (self.run_id,self.correlation_id,self.source_frame_id,self.clock_domain,self.configuration_id)):raise ValueError("missing machine state identity")
        if isinstance(self.timestamp,bool) or not isinstance(self.timestamp,(int,float)) or not math.isfinite(self.timestamp):raise ValueError("invalid machine state timestamp")
        if not isinstance(self.values,tuple) or not all(isinstance(key,str) and key and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) for key,value in self.values):raise ValueError("invalid machine state values")

@dataclass(frozen=True)
class SafetyDecision:
    run_id:str;correlation_id:str;path_id:str;path_hash:str;motion_plan_id:str;motion_plan_hash:str;status:SafetyStatus;reason_codes:tuple[str,...];checked_boundaries:tuple[str,...];safety_config_version:str;timestamp:float;clock_domain:str;schema_version:str=INTEGRATION_SCHEMA_VERSION;execution_authorized:bool=False;limits_hash:str="LIMITS_UNAVAILABLE"
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
            if not isinstance(machine_state,SyntheticMachineState):reasons.append("MACHINE_STATE_MISSING")
            if path.run_id!=motion.run_id or path.correlation_id!=motion.correlation_id:reasons.append("RUN_CORRELATION_MISMATCH")
            if path.config_version!=motion.config_version:reasons.append("PLAN_CONFIG_MISMATCH")
            if motion.source_path_id!=path.path_id or motion.source_path_hash!=path.content_hash:reasons.append("PATH_HASH_MISMATCH")
            if path.coordinate_frame!=motion.coordinate_frame or path.units!=motion.units or path.clock_domain!=motion.clock_domain:reasons.append("FRAME_CLOCK_UNITS_MISMATCH")
            if not path.calibration_id or not path.transform_id:reasons.append("CALIBRATION_TRANSFORM_MISSING")
            if path.has_gap:reasons.append("PATH_GAP")
            if path.qualification_status!="VERIFIED":reasons.append("QUALIFICATION_UNVERIFIED")
            if path.uncertainty_status is not UncertaintyStatus.KNOWN:reasons.append("UNCERTAINTY_UNKNOWN")
            resolver=ProvenanceResolver({item.point_id:item for item in path.point_provenance})
            for point in path.point_ids:reasons.extend(resolver.validate(point).reason_codes)
            for point in path.point_provenance:
                if (point.source_run_id,point.source_frame_id,point.timestamp,point.clock_domain,point.coordinate_frame,point.units,point.calibration_id,point.transform_id,point.config_version)!=(path.run_id,path.source_frame_id,path.timestamp,path.clock_domain,path.coordinate_frame,path.units,path.calibration_id,path.transform_id,path.config_version):reasons.append("POINT_PATH_METADATA_MISMATCH")
            if tuple(motion.target_point_ids)!=path.point_ids:reasons.append("TARGET_PATH_REFERENCE_MISMATCH")
            if tuple(motion.segment_ids)!=tuple(path.segment_ids):reasons.append("SEGMENT_ORDER_MISMATCH")
            if path.path_id not in motion.provenance_references:reasons.append("MOTION_PROVENANCE_REFERENCE_MISSING")
            for index,item in enumerate(motion.timing_records):
                if item.target_index!=index:reasons.append("TIMING_TARGET_INDEX_MISMATCH")
                if item.source_path_id!=path.path_id:reasons.append("TIMING_SOURCE_PATH_MISMATCH")
                if item.segment_id not in path.segment_ids:reasons.append("TIMING_SEGMENT_MISMATCH")
            if any(item.timing_status is TimingStatus.UNSPECIFIED for item in motion.timing_records):reasons.append("TIMING_UNSPECIFIED")
            if isinstance(machine_state,SyntheticMachineState):
                if (machine_state.run_id,machine_state.correlation_id,machine_state.source_frame_id)!=(path.run_id,path.correlation_id,path.source_frame_id):reasons.append("MACHINE_STATE_IDENTITY_MISMATCH")
                if machine_state.clock_domain!=path.clock_domain:reasons.append("MACHINE_STATE_CLOCK_DOMAIN_MISMATCH")
                if machine_state.configuration_id!=limits.config_version:reasons.append("MACHINE_STATE_CONFIG_MISMATCH")
                if machine_state.evidence_status is not MachineStateEvidenceStatus.KNOWN:reasons.append("MACHINE_STATE_EVIDENCE_UNKNOWN")
            xyz=motion.target_coordinates
            if limits.max_path_motion_deviation_m is None:reasons.append("PATH_MOTION_TOLERANCE_UNSPECIFIED")
            elif np.linalg.norm(motion.target_coordinates-path.coordinates,axis=1).max()>limits.max_path_motion_deviation_m:reasons.append("PATH_MOTION_DEVIATION_EXCEEDED")
            if ((path.coordinates<limits.workspace_min)|(path.coordinates>limits.workspace_max)).any():reasons.append("PATH_WORKSPACE_VIOLATION")
            if ((xyz<limits.workspace_min)|(xyz>limits.workspace_max)).any():reasons.append("WORKSPACE_VIOLATION")
            if len(xyz)>1 and (np.linalg.norm(np.diff(xyz,axis=0),axis=1)>limits.max_step_m).any():reasons.append("TARGET_STEP_EXCEEDED")
            if any(item.speed_m_s is not None and item.speed_m_s>limits.max_speed_m_s for item in motion.timing_records):reasons.append("SPEED_LIMIT_EXCEEDED")
            if limits.max_c_step_rad is None:reasons.append("C_STEP_LIMIT_UNSPECIFIED")
            else:
                c=[item.c_rad for item in motion.timing_records]
                if any(value is None for value in c):reasons.append("C_VALUE_MISSING")
                elif len(c)>1 and any(abs(b-a)>limits.max_c_step_rad for a,b in zip(c,c[1:])):reasons.append("C_STEP_LIMIT_EXCEEDED")
            reasons=tuple(dict.fromkeys(reasons));blocked=("MACHINE_STATE_MISSING","CALIBRATION_TRANSFORM_MISSING","TIMING_UNSPECIFIED","PATH_GAP","UNCERTAINTY_UNKNOWN","PATH_MOTION_TOLERANCE_UNSPECIFIED","C_STEP_LIMIT_UNSPECIFIED","C_VALUE_MISSING")
            status=SafetyStatus.ACCEPTED if not reasons else (SafetyStatus.BLOCKED if any(code in reasons for code in blocked) else SafetyStatus.REJECTED)
        except Exception as exc:
            reasons=("SAFETY_VALIDATOR_ERROR",type(exc).__name__);status=SafetyStatus.ERROR
            path=path if isinstance(path,PathPlan) else None;motion=motion if isinstance(motion,MotionPlan) else None
        decision=SafetyDecision(path.run_id if path else "unknown",path.correlation_id if path else "unknown",path.path_id if path else "unknown",path.content_hash if path else "unknown",motion.motion_plan_id if motion else "unknown",motion.content_hash if motion else "unknown",status,reasons,checked,limits.config_version if isinstance(limits,SyntheticMachineLimits) else "unknown",path.timestamp if path else 0.,path.clock_domain if path else "unknown",limits_hash=limits.limits_hash if isinstance(limits,SyntheticMachineLimits) else "LIMITS_UNAVAILABLE")
        event=SafetyAuditEvent(decision.run_id,decision.correlation_id,"independent_safety",decision.path_hash,decision.motion_plan_hash,decision.status,decision.reason_codes,decision.safety_config_version,decision.timestamp,decision.clock_domain)
        audit_complete=True
        if audit_sink is not None:
            try:audit_sink.emit(event)
            except Exception:
                audit_complete=False
                decision=SafetyDecision(decision.run_id,decision.correlation_id,decision.path_id,decision.path_hash,decision.motion_plan_id,decision.motion_plan_hash,SafetyStatus.BLOCKED,tuple(dict.fromkeys(decision.reason_codes+("EVIDENCE_INCOMPLETE",))),checked,decision.safety_config_version,decision.timestamp,decision.clock_domain)
        return decision,audit_complete

def apply_boundary_veto(boundary_id,input_identity,upstream_status,safety_decision=None,*,protected=False,path=None,motion=None,limits=None):
    if upstream_status is not SafetyStatus.ACCEPTED:return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("UPSTREAM_NOT_ACCEPTED",))
    if protected and safety_decision is None:return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("SAFETY_DECISION_MISSING",))
    if protected and (not isinstance(path,PathPlan) or not isinstance(motion,MotionPlan) or not isinstance(limits,SyntheticMachineLimits)):return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("BOUNDARY_INPUT_EVIDENCE_MISSING",))
    if protected and (safety_decision.run_id,safety_decision.correlation_id,safety_decision.path_id,safety_decision.path_hash,safety_decision.motion_plan_id,safety_decision.motion_plan_hash,safety_decision.safety_config_version,safety_decision.limits_hash)!=(path.run_id,path.correlation_id,path.path_id,path.content_hash,motion.motion_plan_id,motion.content_hash,limits.config_version,limits.limits_hash):return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("SAFETY_DECISION_IDENTITY_MISMATCH",))
    if safety_decision is not None and safety_decision.status is not SafetyStatus.ACCEPTED:return BoundaryVeto(boundary_id,input_identity,SafetyStatus.BLOCKED,("SAFETY_VETO",),safety_decision.motion_plan_hash)
    return BoundaryVeto(boundary_id,input_identity,SafetyStatus.ACCEPTED,())
