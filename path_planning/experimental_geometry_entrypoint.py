"""Opt-in, geometry-gated benchmark entrypoint for the Legacy PathPlanner.

This module deliberately separates metric boundary evidence used by the gate
from the legacy planner's pixel/RGB-D inputs.  It never derives a transform
between them.  ``ACCEPT`` authorizes only an experimental benchmark call, and
never grants execution authorization.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import PathConfig
from data_source import CameraCalibration
from path_planning.legacy_geometry_gate import (
    GeometryEvidence,
    GeometryGateResult,
    GeometryGateStatus,
    LegacyGeometryGate,
    LegacyGeometryGateConfig,
)
from path_planning.path_planner import PathPlanner


class BenchmarkStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    CONTEXT_INVALID = "CONTEXT_INVALID"
    SUCCESS = "SUCCESS"
    BENCHMARK_FAILED = "BENCHMARK_FAILED"


def _readonly_copy(value):
    if value is None:
        return None
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeometryEvidenceSnapshot:
    coordinate_frame: str | None
    units: str | None
    timestamp: float | None
    clock_domain: str | None
    source_id: str | None
    topology_verified: bool
    correspondence_verified: bool
    reference_centerline: np.ndarray | None


@dataclass(frozen=True)
class LegacyBenchmarkContext:
    """Explicit legacy pixel/RGB-D inputs plus their declared provenance."""

    left_boundary_pixels: np.ndarray | None = None
    right_boundary_pixels: np.ndarray | None = None
    depth: np.ndarray | None = None
    calibration: CameraCalibration | None = None
    source_id: str | None = None
    timestamp: float | None = None
    clock_domain: str | None = None
    coordinate_frame: str | None = "camera"
    units: str | None = "pixels"
    synthetic: bool = False
    mapping_provenance: str | None = None
    mapping_verified: bool = False


@dataclass(frozen=True)
class LegacyBenchmarkContextSnapshot:
    source_id: str | None
    timestamp: float | None
    clock_domain: str | None
    coordinate_frame: str | None
    units: str | None
    synthetic: bool
    mapping_provenance: str | None
    mapping_verified: bool


@dataclass(frozen=True)
class ExperimentalGeometryResult:
    gate_result: GeometryGateResult
    benchmark_status: BenchmarkStatus
    reason_codes: tuple[str, ...]
    evidence: GeometryEvidenceSnapshot | None
    context: LegacyBenchmarkContextSnapshot | None
    output_path: np.ndarray | None
    experimental: bool = True
    execution_authorized: bool = False


class ExperimentalGeometryEntrypoint:
    """Run Legacy PathPlanner only after an explicit ``ACCEPT`` gate result."""

    def __init__(self, gate_config=None, path_config=None, gate=None, path_planner=None):
        self._gate = gate or LegacyGeometryGate(gate_config or LegacyGeometryGateConfig())
        self._path_planner = path_planner or PathPlanner(path_config or PathConfig())

    def run(self, left_boundary_metric, right_boundary_metric, evidence, context=None):
        """Return a benchmark result without invoking downstream motion/control modules."""
        evidence_snapshot = self._snapshot_evidence(evidence)
        context_snapshot = self._snapshot_context(context)
        if not isinstance(evidence, GeometryEvidence):
            return self._result(
                GeometryGateResult(GeometryGateStatus.UNCERTAIN, ("INVALID_GEOMETRY_EVIDENCE",)),
                BenchmarkStatus.NOT_RUN,
                ("INVALID_GEOMETRY_EVIDENCE",),
                evidence_snapshot,
                context_snapshot,
            )

        gate_result = self._gate.evaluate(left_boundary_metric, right_boundary_metric, evidence)
        if gate_result.status is not GeometryGateStatus.ACCEPT:
            return self._result(
                gate_result, BenchmarkStatus.NOT_RUN, gate_result.reason_codes,
                evidence_snapshot, context_snapshot,
            )

        context_reasons = self._validate_context(context, evidence)
        if context_reasons:
            return self._result(
                gate_result, BenchmarkStatus.CONTEXT_INVALID, tuple(context_reasons),
                evidence_snapshot, context_snapshot,
            )

        try:
            output_path = self._path_planner.plan(
                context.left_boundary_pixels,
                context.right_boundary_pixels,
                context.depth,
                context.calibration,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            return self._result(
                gate_result,
                BenchmarkStatus.BENCHMARK_FAILED,
                ("LEGACY_PATH_BENCHMARK_FAILED", type(error).__name__),
                evidence_snapshot,
                context_snapshot,
            )
        return self._result(
            gate_result, BenchmarkStatus.SUCCESS, (), evidence_snapshot,
            context_snapshot, output_path,
        )

    @staticmethod
    def _result(gate_result, status, reasons, evidence, context, output_path=None):
        return ExperimentalGeometryResult(
            gate_result=gate_result,
            benchmark_status=status,
            reason_codes=tuple(reasons),
            evidence=evidence,
            context=context,
            output_path=_readonly_copy(output_path),
        )

    @staticmethod
    def _snapshot_evidence(evidence):
        if not isinstance(evidence, GeometryEvidence):
            return None
        return GeometryEvidenceSnapshot(
            coordinate_frame=evidence.coordinate_frame,
            units=evidence.units,
            timestamp=evidence.timestamp,
            clock_domain=evidence.clock_domain,
            source_id=evidence.source_id,
            topology_verified=evidence.topology_verified,
            correspondence_verified=evidence.correspondence_verified,
            reference_centerline=_readonly_copy(evidence.reference_centerline),
        )

    @staticmethod
    def _snapshot_context(context):
        if not isinstance(context, LegacyBenchmarkContext):
            return None
        return LegacyBenchmarkContextSnapshot(
            source_id=context.source_id,
            timestamp=context.timestamp,
            clock_domain=context.clock_domain,
            coordinate_frame=context.coordinate_frame,
            units=context.units,
            synthetic=context.synthetic,
            mapping_provenance=context.mapping_provenance,
            mapping_verified=context.mapping_verified,
        )

    @staticmethod
    def _valid_pixels(value):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return False
        return array.ndim == 2 and array.shape[1:] == (2,) and len(array) >= 2 and np.isfinite(array).all()

    def _validate_context(self, context, evidence):
        if not isinstance(context, LegacyBenchmarkContext):
            return ("MISSING_BENCHMARK_CONTEXT",)
        reasons = []
        if not self._valid_pixels(context.left_boundary_pixels) or not self._valid_pixels(context.right_boundary_pixels):
            reasons.append("INVALID_PIXEL_BOUNDARIES")
        depth = np.asarray(context.depth) if context.depth is not None else np.empty(())
        if depth.ndim != 2 or not np.isfinite(depth).all() or (depth <= 0).any():
            reasons.append("INVALID_DEPTH")
        if not isinstance(context.calibration, CameraCalibration):
            reasons.append("INVALID_CAMERA_CALIBRATION")
        if context.coordinate_frame != "camera" or context.units != "pixels":
            reasons.append("INVALID_LEGACY_INPUT_FRAME")
        if not context.mapping_verified or not context.mapping_provenance:
            reasons.append("MISSING_BOUNDARY_MAPPING_PROVENANCE")
        if (context.source_id != evidence.source_id or context.timestamp != evidence.timestamp
                or context.clock_domain != evidence.clock_domain):
            reasons.append("BENCHMARK_CONTEXT_METADATA_MISMATCH")
        return tuple(reasons)
