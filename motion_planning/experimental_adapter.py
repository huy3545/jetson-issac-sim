"""Experimental, non-executable handoff from qualified paths to Motion research.

This module deliberately does not import the V1 MotionPlanner, SafetyMonitor, or
MachineController.  It preserves independently qualified runs as independent
records; it does not generate targets or commands.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from path_planning.path_qualification import PathQualificationStatus, UncertaintyStatus


class HandoffMode(str, Enum):
    STRICT = "strict"
    GEOMETRY_ONLY = "geometry_only"


@dataclass(frozen=True)
class MotionHandoffConstraints:
    coordinate_frame: str | None = None
    units: str | None = "m"
    timestamp: float | None = None
    clock_domain: str | None = None


@dataclass(frozen=True)
class MotionHandoffSegment:
    """An isolated input record for future Motion research, never a command."""
    segment_id: str
    source_id: int
    source_indices: tuple[int, int]
    xyz: np.ndarray
    coordinate_frame: str
    units: str
    timestamp: float
    clock_domain: str
    qualification_status: PathQualificationStatus
    uncertainty_status: UncertaintyStatus
    provenance: tuple[str, str, float, str]
    diagnostics: tuple[str, ...]
    experimental: bool
    execution_authorized: bool = False


@dataclass(frozen=True)
class MotionHandoffResult:
    mode: HandoffMode
    accepted_segments: tuple[MotionHandoffSegment, ...]
    rejected_segments: tuple[MotionHandoffSegment, ...]
    diagnostics: tuple[str, ...] = ()


def _readonly_copy(value):
    copied = np.array(value, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


class ExperimentalMotionAdapter:
    """Filters qualified segments without joining them or authorizing execution."""

    _FATAL_CODES = frozenset({
        "INVALID_XYZ", "INVALID_PROVENANCE", "PROVENANCE_MISMATCH",
        "UNVERIFIED_GEOMETRY", "CONSECUTIVE_DUPLICATE", "STEP_EXCEEDED",
        "Z_JUMP_EXCEEDED", "SURFACE_UNCERTAINTY_EXCEEDED",
    })

    def __init__(self, constraints=MotionHandoffConstraints()):
        self.constraints = constraints

    def handoff(self, result, *, mode=HandoffMode.STRICT):
        mode = HandoffMode(mode)
        accepted, rejected = [], []
        for source in result.segments:
            record, reasons = self._record(source, mode)
            if reasons:
                rejected.append(self._with_diagnostics(record, reasons))
            else:
                accepted.append(record)
        return MotionHandoffResult(mode, tuple(accepted), tuple(rejected))

    def _record(self, source, mode):
        xyz = _readonly_copy(source.xyz)
        diagnostics = tuple(source.reason_codes)
        record = MotionHandoffSegment(
            source.segment_id, source.source_id, tuple(source.source_indices), xyz,
            source.coordinate_frame, source.units, source.timestamp, source.clock_domain,
            source.status, source.uncertainty_status,
            (source.coordinate_frame, source.units, source.timestamp, source.clock_domain),
            diagnostics, mode is HandoffMode.GEOMETRY_ONLY)
        reasons = list(self._metadata_reasons(record))
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 2 or not np.isfinite(xyz).all():
            reasons.append("INVALID_XYZ")
        if any(code in self._FATAL_CODES for code in diagnostics):
            reasons.append("BLOCKING_QUALIFICATION_DIAGNOSTIC")
        if mode is HandoffMode.STRICT:
            if source.status is not PathQualificationStatus.VERIFIED:
                reasons.append("QUALIFICATION_NOT_VERIFIED")
            if source.uncertainty_status is not UncertaintyStatus.KNOWN:
                reasons.append("UNCERTAINTY_NOT_VERIFIED")
            if diagnostics:
                reasons.append("QUALIFICATION_DIAGNOSTIC_PRESENT")
        elif source.status is PathQualificationStatus.INVALID:
            reasons.append("QUALIFICATION_INVALID")
        return record, tuple(dict.fromkeys(reasons))

    def _metadata_reasons(self, record):
        expected = self.constraints
        reasons = []
        if not record.coordinate_frame or record.units != "m" or not record.clock_domain or not np.isfinite(record.timestamp):
            reasons.append("INVALID_PROVENANCE")
        if expected.coordinate_frame is not None and record.coordinate_frame != expected.coordinate_frame:
            reasons.append("FRAME_MISMATCH")
        if expected.units is not None and record.units != expected.units:
            reasons.append("UNITS_MISMATCH")
        if expected.timestamp is not None and record.timestamp != expected.timestamp:
            reasons.append("TIMESTAMP_MISMATCH")
        if expected.clock_domain is not None and record.clock_domain != expected.clock_domain:
            reasons.append("CLOCK_DOMAIN_MISMATCH")
        return reasons

    @staticmethod
    def _with_diagnostics(record, reasons):
        return MotionHandoffSegment(
            record.segment_id, record.source_id, record.source_indices, record.xyz,
            record.coordinate_frame, record.units, record.timestamp, record.clock_domain,
            record.qualification_status, record.uncertainty_status, record.provenance,
            tuple(dict.fromkeys(record.diagnostics + tuple(reasons))), record.experimental)
