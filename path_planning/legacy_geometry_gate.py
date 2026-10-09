"""Experimental evidence-based preflight for Legacy PathPlanner inputs.

It does not alter ``PathPlanner.plan`` or create a fallback path.  ACCEPT means
only that the supplied open-channel contract passed these limited checks.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class GeometryGateStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class GeometryEvidence:
    coordinate_frame: str | None = None
    units: str | None = None
    timestamp: float | None = None
    clock_domain: str | None = None
    source_id: str | None = None
    topology_verified: bool = False
    correspondence_verified: bool = False
    reference_centerline: np.ndarray | None = None


@dataclass(frozen=True)
class LegacyGeometryGateConfig:
    resample_points: int = 128
    numerical_zero_length_m: float = 1e-9
    correspondence_ambiguity_m: float = 1e-6
    max_reference_deviation_m: float | None = None
    max_reference_rms_m: float | None = None


@dataclass(frozen=True)
class GeometryGateResult:
    status: GeometryGateStatus
    reason_codes: tuple[str, ...]
    same_endpoint_cost_m: float | None = None
    reverse_endpoint_cost_m: float | None = None
    max_reference_deviation_m: float | None = None
    rms_reference_deviation_m: float | None = None


def _cross(a, b, c):
    u, v = b-a, c-a
    return float(u[0]*v[1] - u[1]*v[0])


def _proper_intersection(a, b, c, d, tolerance=1e-12):
    return _cross(a, b, c)*_cross(a, b, d) < -tolerance and _cross(c, d, a)*_cross(c, d, b) < -tolerance


def _self_intersects(polyline):
    for index in range(len(polyline)-1):
        for other in range(index+2, len(polyline)-1):
            if _proper_intersection(polyline[index], polyline[index+1], polyline[other], polyline[other+1]):
                return True
    return False


def _lines_intersect(first, second):
    return any(_proper_intersection(a, b, c, d)
               for a, b in zip(first[:-1], first[1:])
               for c, d in zip(second[:-1], second[1:]))


def _polygon_intersects(left, right):
    polygon = np.vstack((left, right[::-1]))
    count = len(polygon)
    for index in range(count):
        for other in range(index+1, count):
            if other == index+1 or (index == 0 and other == count-1):
                continue
            if _proper_intersection(polygon[index], polygon[(index+1) % count], polygon[other], polygon[(other+1) % count]):
                return True
    return False


def _resample(line, count):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    samples = np.linspace(0., distance[-1], count)
    return np.column_stack([np.interp(samples, distance, line[:, axis]) for axis in range(2)])


def _point_polyline_distance(points, line):
    values = []
    for point in points:
        start, end = line[:-1], line[1:]
        vector = end-start
        projection = np.clip(((point-start)*vector).sum(axis=1)/(vector*vector).sum(axis=1), 0., 1.)
        values.append(np.linalg.norm(start+projection[:, None]*vector-point, axis=1).min())
    return np.asarray(values)


class LegacyGeometryGate:
    """Opt-in structural gate; unknown evidence never becomes ACCEPT."""

    def __init__(self, config=LegacyGeometryGateConfig()):
        self.config = config

    def evaluate(self, left_boundary, right_boundary, evidence=GeometryEvidence()):
        left, right, reasons = self._validate(left_boundary, right_boundary)
        if reasons:
            return GeometryGateResult(GeometryGateStatus.REJECT, tuple(reasons))
        if _self_intersects(left) or _self_intersects(right):
            return GeometryGateResult(GeometryGateStatus.REJECT, ("BOUNDARY_SELF_INTERSECTION",))

        same = float(np.linalg.norm(left[0]-right[0]) + np.linalg.norm(left[-1]-right[-1]))
        reverse = float(np.linalg.norm(left[0]-right[-1]) + np.linalg.norm(left[-1]-right[0]))
        if abs(same-reverse) <= self.config.correspondence_ambiguity_m:
            return GeometryGateResult(GeometryGateStatus.UNCERTAIN, ("AMBIGUOUS_ENDPOINT_CORRESPONDENCE",), same, reverse)
        if reverse < same:
            right = right[::-1]
        if _lines_intersect(left, right) or _polygon_intersects(left, right):
            return GeometryGateResult(GeometryGateStatus.REJECT, ("CHANNEL_SELF_INTERSECTION",), same, reverse)

        uncertain = self._evidence_reasons(evidence)
        reference_metrics = self._reference_metrics(left, right, evidence.reference_centerline)
        if reference_metrics is None:
            uncertain.append("MISSING_REFERENCE_CENTERLINE")
            maximum = rms = None
        else:
            maximum, rms = reference_metrics
            if self.config.max_reference_deviation_m is None or self.config.max_reference_rms_m is None:
                uncertain.append("REFERENCE_THRESHOLDS_UNCONFIGURED")
            elif maximum > self.config.max_reference_deviation_m or rms > self.config.max_reference_rms_m:
                return GeometryGateResult(GeometryGateStatus.REJECT, ("REFERENCE_DEVIATION_EXCEEDED",), same, reverse, maximum, rms)
        if uncertain:
            return GeometryGateResult(GeometryGateStatus.UNCERTAIN, tuple(sorted(set(uncertain))), same, reverse, maximum, rms)
        return GeometryGateResult(GeometryGateStatus.ACCEPT, (), same, reverse, maximum, rms)

    def _validate(self, left, right):
        reasons = []
        try:
            left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        except (TypeError, ValueError):
            return None, None, ["INVALID_BOUNDARY_SHAPE"]
        for boundary in (left, right):
            if boundary.ndim != 2 or boundary.shape[1:] != (2,) or len(boundary) < 2:
                reasons.append("INVALID_BOUNDARY_SHAPE")
            elif not np.isfinite(boundary).all():
                reasons.append("NONFINITE_BOUNDARY")
            elif (np.linalg.norm(np.diff(boundary, axis=0), axis=1) <= self.config.numerical_zero_length_m).any():
                reasons.append("DEGENERATE_BOUNDARY_SEGMENT")
        return left, right, reasons

    def _evidence_reasons(self, evidence):
        reasons = []
        if not evidence.coordinate_frame or evidence.units != "m" or evidence.timestamp is None or not np.isfinite(evidence.timestamp) or not evidence.clock_domain or not evidence.source_id:
            reasons.append("MISSING_GEOMETRY_METADATA")
        if not evidence.topology_verified:
            reasons.append("TOPOLOGY_UNVERIFIED")
        if not evidence.correspondence_verified:
            reasons.append("CORRESPONDENCE_UNVERIFIED")
        return reasons

    def _reference_metrics(self, left, right, reference):
        if reference is None:
            return None
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1:] != (2,) or len(reference) < 2 or not np.isfinite(reference).all():
            return None
        midpoint = (_resample(left, self.config.resample_points) + _resample(right, self.config.resample_points))/2
        distance = _point_polyline_distance(midpoint, reference)
        return float(distance.max()), float(np.sqrt(np.mean(distance*distance)))
