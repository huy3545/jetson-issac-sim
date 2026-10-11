"""Fail-closed provenance and compatibility validation for Stage 7B-1A."""
from dataclasses import dataclass
import math

from .contracts import DerivationMethod, PointProvenance, ProvenanceObservationStatus

@dataclass(frozen=True)
class ValidationOutcome:
    accepted: bool
    reason_codes: tuple[str, ...] = ()

class ProvenanceResolver:
    """Recursively resolves input point IDs; it never treats absent parents as observed."""
    def __init__(self, points):
        if not isinstance(points, dict) or not all(isinstance(key, str) and isinstance(value, PointProvenance) for key,value in points.items()):
            raise ValueError("points must map point IDs to PointProvenance")
        self.points = dict(points)

    def validate(self, point_id):
        reasons = self._visit(point_id, (), None)
        return ValidationOutcome(not reasons, tuple(dict.fromkeys(reasons)))

    def _visit(self, point_id, stack, expected_run):
        point = self.points.get(point_id)
        if point is None:return ["SOURCE_PROVENANCE_UNRESOLVED"]
        if point_id in stack:return ["PROVENANCE_CYCLE"]
        reasons = list(point.strict_reason_codes())
        if expected_run is not None and point.source_run_id != expected_run:reasons.append("SOURCE_RUN_MISMATCH")
        if point.derivation_method is DerivationMethod.OBSERVATION:
            return reasons
        for source_id in point.source_point_ids:
            source = self.points.get(source_id)
            if source is None:
                reasons.append("SOURCE_PROVENANCE_UNRESOLVED");continue
            if source.observation_status is ProvenanceObservationStatus.PREDICTED:reasons.append("SOURCE_PREDICTED")
            if source.observation_status is ProvenanceObservationStatus.UNKNOWN:reasons.append("SOURCE_OBSERVATION_UNKNOWN")
            reasons.extend(self._visit(source_id, stack+(point_id,), point.source_run_id))
        return reasons

class CompatibilityValidator:
    """Compares boundary metadata without assuming a transform or sync tolerance."""
    FIELDS = (("source_run_id","SOURCE_RUN_MISMATCH"),("clock_domain","CLOCK_DOMAIN_MISMATCH"),
              ("coordinate_frame","COORDINATE_FRAME_MISMATCH"),("units","UNITS_MISMATCH"),
              ("config_version","CONFIG_VERSION_MISMATCH"))
    def compare(self, left, right, *, require_segment=False, timestamp_tolerance_s=None):
        if not isinstance(left,PointProvenance) or not isinstance(right,PointProvenance):
            raise ValueError("compatibility requires PointProvenance")
        if timestamp_tolerance_s is not None and (isinstance(timestamp_tolerance_s,bool) or not isinstance(timestamp_tolerance_s,(int,float)) or not math.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s<0):
            raise ValueError("invalid timestamp_tolerance_s")
        reasons=[]
        for field,code in self.FIELDS:
            if getattr(left,field)!=getattr(right,field):reasons.append(code)
        if (left.source_frame_id,left.timestamp)!=(right.source_frame_id,right.timestamp):
            if left.source_frame_id!=right.source_frame_id or timestamp_tolerance_s is None or abs(left.timestamp-right.timestamp)>timestamp_tolerance_s:reasons.append("FRAME_TIME_MISMATCH")
        if (left.calibration_id,left.transform_id)!=(right.calibration_id,right.transform_id):reasons.append("CALIBRATION_TRANSFORM_MISMATCH")
        if require_segment and left.segment_id!=right.segment_id:reasons.append("SEGMENT_IDENTITY_MISMATCH")
        return ValidationOutcome(not reasons,tuple(dict.fromkeys(reasons)))
