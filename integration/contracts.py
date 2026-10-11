"""Versioned immutable structures for experimental integration only."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math
import re

INTEGRATION_SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = frozenset((INTEGRATION_SCHEMA_VERSION,))
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

class AdmissionStatus(str, Enum):
    ACCEPTED="ACCEPTED"; REJECTED="REJECTED"; BLOCKED="BLOCKED"; ERROR="ERROR"
class ProvenanceObservationStatus(str, Enum):
    OBSERVED="OBSERVED"; PREDICTED="PREDICTED"; UNKNOWN="UNKNOWN"
class ConfidenceStatus(str, Enum):
    VERIFIED="VERIFIED"; UNKNOWN="UNKNOWN"
class DerivationMethod(str, Enum):
    OBSERVATION="OBSERVATION"; INTERPOLATION="INTERPOLATION"; RESAMPLING="RESAMPLING"; SMOOTHING="SMOOTHING"; MIDPOINT="MIDPOINT"; COORDINATE_TRANSFORM="COORDINATE_TRANSFORM"
class TimingStatus(str, Enum):
    SPECIFIED="SPECIFIED"; UNSPECIFIED="UNSPECIFIED"
class ConfigurationKind(str, Enum):
    SYNTHETIC="SYNTHETIC"; UNSPECIFIED="UNSPECIFIED"

def _identifier(value, name, required=False):
    if value is None:
        if required: raise ValueError(f"{name} is required")
        return
    if not isinstance(value,(str,int)) or not _IDENTIFIER.fullmatch(str(value)): raise ValueError(f"malformed {name}")
def _finite(value,name):
    if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)): raise ValueError(f"{name} must be finite when supplied")
def _schema(value):
    if value not in SUPPORTED_SCHEMA_VERSIONS: raise ValueError(f"unsupported schema_version: {value}")

@dataclass(frozen=True)
class PointProvenance:
    point_id:str; source_point_ids:tuple[str,...]; source_run_id:str|None; source_frame_id:str|int|None; segment_id:str|None
    observation_status:ProvenanceObservationStatus; confidence_status:ConfidenceStatus; confidence_value:float|None; uncertainty_status:str
    timestamp:float|None; clock_domain:str|None; coordinate_frame:str|None; units:str|None; calibration_id:str|None; transform_id:str|None
    config_version:str|None; derivation_method:DerivationMethod; schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version);_identifier(self.point_id,"point_id",True)
        if not isinstance(self.observation_status,ProvenanceObservationStatus):raise ValueError("unknown observation_status")
        if not isinstance(self.confidence_status,ConfidenceStatus):raise ValueError("unknown confidence_status")
        if not isinstance(self.derivation_method,DerivationMethod):raise ValueError("unknown derivation_method")
        if not isinstance(self.source_point_ids,tuple) or not self.source_point_ids:raise ValueError("source_point_ids are required")
        for source in self.source_point_ids:_identifier(source,"source_point_id",True)
        for name in ("source_run_id","source_frame_id","segment_id","calibration_id","transform_id","config_version"):_identifier(getattr(self,name),name)
        _finite(self.timestamp,"timestamp");_finite(self.confidence_value,"confidence_value")
        if self.confidence_value is not None and not 0<=self.confidence_value<=1:raise ValueError("confidence_value must be in [0, 1]")
        if not isinstance(self.uncertainty_status,str) or not self.uncertainty_status:raise ValueError("uncertainty_status is required")
    def strict_reason_codes(self,source_observations=()):
        required=("source_run_id","source_frame_id","segment_id","timestamp","clock_domain","coordinate_frame","units","calibration_id","transform_id","config_version")
        reasons=[f"MISSING_{name.upper()}" for name in required if getattr(self,name) in (None,"")]
        if self.observation_status is not ProvenanceObservationStatus.OBSERVED:reasons.append("POINT_NOT_OBSERVED")
        if self.confidence_status is not ConfidenceStatus.VERIFIED:reasons.append("CONFIDENCE_UNKNOWN")
        if self.uncertainty_status!="KNOWN":reasons.append("UNCERTAINTY_UNKNOWN")
        if any(item is not ProvenanceObservationStatus.OBSERVED for item in source_observations):reasons.append("DERIVED_FROM_NON_OBSERVED_SOURCE")
        return tuple(dict.fromkeys(reasons))

@dataclass(frozen=True)
class MotionTimingRecord:
    target_index:int;segment_id:str;source_path_id:str;coordinate_frame:str;units:str;timestamp:float|None;duration_s:float|None;speed_m_s:float|None;c_rad:float|None;admission_status:AdmissionStatus;reason_codes:tuple[str,...]=();schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version)
        if not isinstance(self.admission_status,AdmissionStatus):raise ValueError("unknown admission_status")
        if not isinstance(self.target_index,int) or self.target_index<0:raise ValueError("target_index must be a non-negative integer")
        for name in ("segment_id","source_path_id","coordinate_frame","units"):_identifier(getattr(self,name),name,True)
        for name in ("timestamp","duration_s","speed_m_s","c_rad"):_finite(getattr(self,name),name)
        if self.duration_s is not None and self.duration_s<0:raise ValueError("duration_s cannot be negative")
    @property
    def timing_status(self):return TimingStatus.SPECIFIED if self.timestamp is not None or self.duration_s is not None else TimingStatus.UNSPECIFIED
    def timing_reason_codes(self):return () if self.timing_status is TimingStatus.SPECIFIED else ("TIMING_UNSPECIFIED",)

@dataclass(frozen=True)
class SyntheticConfiguration:
    synthetic_camera:str|None;synthetic_calibration:str|None;synthetic_transform:str|None;synthetic_cut_tolerance_mm:float|None;kind:ConfigurationKind;real_accuracy_claim:str="NONE";schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version)
        if not isinstance(self.kind,ConfigurationKind):raise ValueError("unknown configuration kind")
        if self.real_accuracy_claim!="NONE":raise ValueError("real_accuracy_claim must be NONE")
        for name in ("synthetic_camera","synthetic_calibration","synthetic_transform"):_identifier(getattr(self,name),name)
        _finite(self.synthetic_cut_tolerance_mm,"synthetic_cut_tolerance_mm")
        if self.synthetic_cut_tolerance_mm is not None and self.synthetic_cut_tolerance_mm<0:raise ValueError("synthetic_cut_tolerance_mm cannot be negative")
        if self.kind is ConfigurationKind.SYNTHETIC and not all((self.synthetic_camera,self.synthetic_calibration,self.synthetic_transform)):raise ValueError("synthetic configuration identity is incomplete")
        if self.kind is ConfigurationKind.UNSPECIFIED and any((self.synthetic_camera,self.synthetic_calibration,self.synthetic_transform,self.synthetic_cut_tolerance_mm)):raise ValueError("UNSPECIFIED configuration cannot contain synthetic values")
    def strict_reason_codes(self):return () if self.kind is ConfigurationKind.SYNTHETIC else ("REAL_CONFIGURATION_UNSPECIFIED",)

@dataclass(frozen=True)
class IntegrationRecord:
    run_id:str;correlation_id:str;stage_id:str;admission_status:AdmissionStatus;source_frame_id:str|int|None=None;segment_id:str|None=None;timestamp:float|None=None;clock_domain:str|None=None;coordinate_frame:str|None=None;units:str|None=None;config_version:str|None=None;provenance_references:tuple[str,...]=();confidence:float|None=None;geometry_status:str|None=None;uncertainty_status:str|None=None;qualification_status:str|None=None;reason_codes:tuple[str,...]=();diagnostics:tuple[str,...]=();execution_authorized:bool=False;schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version);_identifier(self.run_id,"run_id",True);_identifier(self.correlation_id,"correlation_id",True);_identifier(self.stage_id,"stage_id",True);_identifier(self.source_frame_id,"source_frame_id");_identifier(self.segment_id,"segment_id")
        if not isinstance(self.admission_status,AdmissionStatus):raise ValueError("unknown admission_status")
        _finite(self.timestamp,"timestamp");_finite(self.confidence,"confidence")
        if self.confidence is not None and not 0<=self.confidence<=1:raise ValueError("confidence must be in [0, 1]")
        if self.execution_authorized:raise ValueError("experimental integration cannot authorize execution")

@dataclass(frozen=True)
class IntegrationEvent:
    run_id:str;correlation_id:str;stage_id:str;event_type:str;timestamp:float|None;reason_codes:tuple[str,...]=();diagnostics:tuple[str,...]=();schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version);_identifier(self.run_id,"run_id",True);_identifier(self.correlation_id,"correlation_id",True);_identifier(self.stage_id,"stage_id",True);_finite(self.timestamp,"timestamp")

@dataclass(frozen=True)
class IntegrationRunResult:
    run_id:str;correlation_id:str;records:tuple[IntegrationRecord,...];events:tuple[IntegrationEvent,...];final_status:AdmissionStatus;audit_complete:bool;execution_authorized:bool=False;details:tuple[tuple[str,Any],...]=field(default_factory=tuple);schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        _schema(self.schema_version);_identifier(self.run_id,"run_id",True);_identifier(self.correlation_id,"correlation_id",True)
        if not isinstance(self.final_status,AdmissionStatus):raise ValueError("unknown final_status")
        if self.execution_authorized:raise ValueError("experimental integration cannot authorize execution")
