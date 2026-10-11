"""Experimental segment-isolated reuse of the V1 MotionPlanner.

This layer is deliberately non-executable: it creates research motion records,
never Safety authorization or controller commands.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

from config import MotionConfig
from .experimental_adapter import HandoffMode, MotionHandoffResult
from .motion_planner import MotionPlanner
from path_planning.path_qualification import PathQualificationStatus, UncertaintyStatus


@dataclass(frozen=True)
class PlannedMotionSegment:
    segment_id: str
    source_id: int
    source_indices: tuple[int, int]
    coordinate_frame: str
    units: str
    timestamp: float
    clock_domain: str
    qualification_status: PathQualificationStatus
    uncertainty_status: UncertaintyStatus
    source_provenance: tuple[str, str, float, str]
    targets: tuple
    diagnostics: tuple[str, ...]
    experimental: bool = True
    execution_authorized: bool = False


@dataclass(frozen=True)
class RejectedMotionSegment:
    segment_id: str
    source_id: int
    source_indices: tuple[int, int]
    coordinate_frame: str
    units: str
    timestamp: float
    clock_domain: str
    qualification_status: PathQualificationStatus
    uncertainty_status: UncertaintyStatus
    source_provenance: tuple[str, str, float, str]
    diagnostics: tuple[str, ...]
    experimental: bool = True
    execution_authorized: bool = False


@dataclass(frozen=True)
class SegmentMotionPlanningResult:
    planned_segments: tuple[PlannedMotionSegment, ...]
    rejected_segments: tuple[RejectedMotionSegment, ...]
    diagnostics: tuple[str, ...] = ()
    experimental: bool = True
    execution_authorized: bool = False


class SegmentMotionPlanner:
    """Calls the V1 planner once per strict handoff segment, never across gaps."""

    def __init__(self, config: MotionConfig):
        self._planner = MotionPlanner(config)

    def plan(self, handoff: MotionHandoffResult) -> SegmentMotionPlanningResult:
        planned, rejected = [], []
        if handoff.mode is not HandoffMode.STRICT:
            return SegmentMotionPlanningResult(
                (), tuple(self._rejected(segment, ("STRICT_HANDOFF_REQUIRED",))
                          for segment in handoff.accepted_segments),
                ("STRICT_HANDOFF_REQUIRED",))
        rejected.extend(self._rejected(segment, ("HANDOFF_REJECTED",))
                        for segment in handoff.rejected_segments)
        for segment in handoff.accepted_segments:
            reasons = self._admission_reasons(segment)
            if reasons:
                rejected.append(self._rejected(segment, reasons))
                continue
            try:
                targets = self._planner.plan(segment.xyz)
                reasons = self._target_reasons(segment.xyz, targets)
            except (TypeError, ValueError, FloatingPointError) as error:
                reasons = ("MOTION_PLANNER_REJECTED", type(error).__name__)
            if reasons:
                rejected.append(self._rejected(segment, reasons))
            else:
                planned.append(self._planned(segment, targets))
        return SegmentMotionPlanningResult(tuple(planned), tuple(rejected))

    @staticmethod
    def _admission_reasons(segment):
        xyz = np.asarray(segment.xyz, dtype=float)
        reasons = []
        if segment.experimental or segment.execution_authorized:
            reasons.append("NON_STRICT_HANDOFF")
        if segment.qualification_status is not PathQualificationStatus.VERIFIED:
            reasons.append("QUALIFICATION_NOT_VERIFIED")
        if segment.uncertainty_status is not UncertaintyStatus.KNOWN:
            reasons.append("UNCERTAINTY_NOT_VERIFIED")
        if segment.diagnostics:
            reasons.append("HANDOFF_DIAGNOSTIC_PRESENT")
        if (not segment.coordinate_frame or segment.units != "m" or not segment.clock_domain
                or not np.isfinite(segment.timestamp)):
            reasons.append("INVALID_PROVENANCE")
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 2 or not np.isfinite(xyz).all():
            reasons.append("INVALID_XYZ")
            return tuple(reasons)
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if (ds <= 1e-9).any():
            reasons.append("CONSECUTIVE_DUPLICATE")
            return tuple(reasons)
        s = np.r_[0., np.cumsum(ds)]
        tangent = np.gradient(xyz, s, axis=0)
        if (np.linalg.norm(tangent[:, :2], axis=1) <= 1e-9).any():
            reasons.append("XY_TANGENT_DEGENERATE")
        return tuple(reasons)

    @staticmethod
    def _target_reasons(xyz, targets):
        if len(targets) != len(xyz):
            return ("TARGET_COUNT_MISMATCH",)
        try:
            values = np.array([[target[key] for key in ("x", "y", "z", "c", "speed")]
                               for target in targets], dtype=float)
        except (KeyError, TypeError, ValueError):
            return ("INVALID_TARGET_CONTRACT",)
        if not np.isfinite(values).all() or (values[:, 4] < 0).any():
            return ("INVALID_TARGET_VALUES",)
        if not np.allclose(values[:, :3], xyz, rtol=0., atol=1e-12):
            return ("TARGET_XYZ_MISMATCH",)
        if values[0, 4] != 0. or values[-1, 4] != 0.:
            return ("BOUNDARY_SPEED_NOT_ZERO",)
        return ()

    @staticmethod
    def _planned(segment, targets):
        copied_targets = tuple(MappingProxyType(dict(target)) for target in targets)
        return PlannedMotionSegment(
            segment.segment_id, segment.source_id, tuple(segment.source_indices),
            segment.coordinate_frame, segment.units, segment.timestamp, segment.clock_domain,
            segment.qualification_status, segment.uncertainty_status, tuple(segment.provenance),
            copied_targets, tuple(segment.diagnostics))

    @staticmethod
    def _rejected(segment, reasons):
        return RejectedMotionSegment(
            segment.segment_id, segment.source_id, tuple(segment.source_indices),
            segment.coordinate_frame, segment.units, segment.timestamp, segment.clock_domain,
            segment.qualification_status, segment.uncertainty_status, tuple(segment.provenance),
            tuple(dict.fromkeys(tuple(segment.diagnostics) + tuple(reasons))))
