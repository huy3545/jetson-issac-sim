"""Experimental, evidence-gated Vision pixel to metric projection.

This is deliberately separate from the Legacy PathPlanner.  A numerically valid
projection is not calibration verification, therefore this Stage 6B adapter
never produces ``VERIFIED`` geometry or ``KNOWN`` measurement uncertainty.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np

from vision.contracts import ObservationStatus, OrientationSource, TrackingState, VisionResult


class TransformValidity(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    DECLARED = "DECLARED"
    VERIFIED = "VERIFIED"
    INVALID = "INVALID"


class MetricGeometryStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    INVALID = "INVALID"


class MetricUncertaintyStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    EXCEEDED = "EXCEEDED"


def _readonly(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TransformEvidence:
    camera_frame: str
    reference_frame: str
    units: str
    intrinsics_id: str
    intrinsics_version: str
    extrinsics_id: str
    extrinsics_version: str
    calibration_source: str
    transform_source: str
    calibration_timestamp: float
    transform_timestamp: float
    observation_timestamp: float
    clock_domain: str
    source_id: str
    transform_matrix: np.ndarray
    transform_validity: TransformValidity

    def __post_init__(self):
        object.__setattr__(self, "transform_matrix", _readonly(self.transform_matrix))


@dataclass(frozen=True)
class MetricGeometrySegment:
    name: str
    xyz: np.ndarray | None
    source_id: str
    segment_id: str | None
    run_id: int | None
    source_indices: tuple[int, int] | None
    status: MetricGeometryStatus
    uncertainty_status: MetricUncertaintyStatus
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if self.xyz is not None:
            object.__setattr__(self, "xyz", _readonly(self.xyz))


@dataclass(frozen=True)
class MetricGeometryResult:
    segments: tuple[MetricGeometrySegment, ...]
    source_id: str | None
    camera_frame: str | None
    reference_frame: str | None
    units: str
    timestamp: float | None
    clock_domain: str | None
    transform_evidence: TransformEvidence | None
    calibration_evidence: TransformEvidence | None
    status: MetricGeometryStatus
    uncertainty_status: MetricUncertaintyStatus
    diagnostics: tuple[str, ...] = ()
    experimental: bool = True
    execution_authorized: bool = False


@dataclass(frozen=True)
class ProjectionConfig:
    rotation_atol: float = 1e-8
    homogeneous_atol: float = 1e-10
    max_evidence_age_s: float | None = None


class ExperimentalVisionMetricProjectionAdapter:
    """Project only fully observed registered RGB-D samples with explicit evidence."""

    def __init__(self, config=ProjectionConfig()):
        self.config = config

    def project(self, vision: VisionResult, depth, calibration, evidence: TransformEvidence | None):
        diagnostics = self._admission_diagnostics(vision, depth, calibration, evidence)
        if diagnostics:
            return self._result(vision, evidence, MetricGeometryStatus.INVALID, tuple(diagnostics))

        # Evidence labels, including caller-provided VERIFIED, are not independent
        # calibration proof.  The output consequently remains UNCERTAIN/UNKNOWN.
        segments = []
        for name, line in (("left", vision.left), ("right", vision.right)):
            xyz, reasons = self._project_line(line.points, depth, calibration, evidence.transform_matrix)
            if reasons:
                return self._result(vision, evidence, MetricGeometryStatus.INVALID, tuple(reasons))
            segments.append(MetricGeometrySegment(
                name=name, xyz=xyz, source_id=evidence.source_id, segment_id=None,
                run_id=None, source_indices=(0, len(xyz)),
                status=MetricGeometryStatus.UNCERTAIN,
                uncertainty_status=MetricUncertaintyStatus.UNKNOWN,
                diagnostics=("CALIBRATION_UNVERIFIED", "XYZ_UNCERTAINTY_UNKNOWN"),
            ))
        return MetricGeometryResult(
            segments=tuple(segments), source_id=evidence.source_id,
            camera_frame=evidence.camera_frame, reference_frame=evidence.reference_frame,
            units="m", timestamp=vision.frame_timestamp, clock_domain=vision.frame_clock_domain,
            transform_evidence=evidence, calibration_evidence=evidence,
            status=MetricGeometryStatus.UNCERTAIN,
            uncertainty_status=MetricUncertaintyStatus.UNKNOWN,
            diagnostics=("CALIBRATION_UNVERIFIED", "XYZ_UNCERTAINTY_UNKNOWN"),
        )

    def _result(self, vision, evidence, status, diagnostics):
        source = evidence.source_id if isinstance(evidence, TransformEvidence) else None
        return MetricGeometryResult(
            segments=(), source_id=source,
            camera_frame=evidence.camera_frame if isinstance(evidence, TransformEvidence) else None,
            reference_frame=evidence.reference_frame if isinstance(evidence, TransformEvidence) else None,
            units="m", timestamp=vision.frame_timestamp if isinstance(vision, VisionResult) else None,
            clock_domain=vision.frame_clock_domain if isinstance(vision, VisionResult) else None,
            transform_evidence=evidence if isinstance(evidence, TransformEvidence) else None,
            calibration_evidence=evidence if isinstance(evidence, TransformEvidence) else None,
            status=status, uncertainty_status=MetricUncertaintyStatus.UNKNOWN,
            diagnostics=tuple(sorted(set(diagnostics))),
        )

    def _admission_diagnostics(self, vision, depth, calibration, evidence):
        reasons = []
        if not isinstance(vision, VisionResult):
            return ("INVALID_VISION_RESULT",)
        if not isinstance(evidence, TransformEvidence):
            return ("MISSING_TRANSFORM_EVIDENCE",)
        if vision.coordinate_frame != evidence.camera_frame or vision.units != "pixels":
            reasons.append("VISION_FRAME_OR_UNITS_MISMATCH")
        if (vision.frame_timestamp is None or not np.isfinite(vision.frame_timestamp)
                or vision.frame_timestamp != evidence.observation_timestamp
                or vision.frame_clock_domain != evidence.clock_domain):
            reasons.append("OBSERVATION_TIME_DOMAIN_MISMATCH")
        if vision.frame_id is None or str(vision.frame_id) != evidence.source_id:
            reasons.append("SOURCE_ID_MISMATCH")
        for line, name in ((vision.left, "LEFT"), (vision.right, "RIGHT")):
            reasons.extend(self._line_diagnostics(line, name))
        reasons.extend(self._evidence_diagnostics(evidence))
        try:
            values = np.asarray([calibration.fx, calibration.fy, calibration.cx, calibration.cy], dtype=float)
        except (AttributeError, TypeError, ValueError):
            values = np.array([np.nan])
        if len(values) != 4 or not np.isfinite(values).all() or values[0] <= 0 or values[1] <= 0:
            reasons.append("INVALID_INTRINSICS")
        if vision.camera_to_reference is None or not np.array_equal(np.asarray(vision.camera_to_reference), evidence.transform_matrix):
            reasons.append("VISION_TRANSFORM_MISMATCH")
        array = np.asarray(depth) if depth is not None else np.empty(())
        if array.ndim != 2:
            reasons.append("INVALID_DEPTH_IMAGE")
        return reasons

    @staticmethod
    def _line_diagnostics(line, name):
        reasons = []
        points = line.points
        try:
            valid_points = np.asarray(points, dtype=float)
        except (TypeError, ValueError):
            valid_points = np.empty((0,))
        if (line.observation_status is not ObservationStatus.OBSERVED
                or line.tracking_state is not TrackingState.TRACKING or not line.verified
                or line.prediction_uncertainty_px != 0 or line.missing_ranges
                or line.orientation_source is OrientationSource.AMBIGUOUS):
            reasons.append(f"{name}_NOT_ADMISSIBLE")
        if valid_points.ndim != 2 or valid_points.shape[1:] != (2,) or len(valid_points) < 2 or not np.isfinite(valid_points).all():
            reasons.append(f"{name}_INVALID_PIXELS")
        depth_valid = line.depth_valid
        if depth_valid is None or len(depth_valid) != len(valid_points) or not np.all(depth_valid):
            reasons.append(f"{name}_DEPTH_VALIDITY_UNPROVEN")
        return reasons

    def _evidence_diagnostics(self, evidence):
        required_text = (
            evidence.camera_frame, evidence.reference_frame, evidence.units,
            evidence.intrinsics_id, evidence.intrinsics_version, evidence.extrinsics_id,
            evidence.extrinsics_version, evidence.calibration_source,
            evidence.transform_source, evidence.clock_domain, evidence.source_id,
        )
        reasons = []
        if evidence.units != "m" or not all(required_text):
            reasons.append("MISSING_CALIBRATION_EVIDENCE")
        timestamps = np.asarray((evidence.calibration_timestamp, evidence.transform_timestamp,
                                 evidence.observation_timestamp), dtype=float)
        if not np.isfinite(timestamps).all():
            reasons.append("INVALID_EVIDENCE_TIMESTAMP")
        if self.config.max_evidence_age_s is not None and np.isfinite(timestamps).all():
            if (np.any(timestamps[:2] > timestamps[2])
                    or np.any(timestamps[2] - timestamps[:2] > self.config.max_evidence_age_s)):
                reasons.append("EVIDENCE_TIMESTAMP_POLICY_FAILED")
        if evidence.transform_validity is TransformValidity.INVALID:
            reasons.append("TRANSFORM_DECLARED_INVALID")
        matrix = evidence.transform_matrix
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return reasons + ["INVALID_TRANSFORM_MATRIX"]
        if not np.allclose(matrix[3], (0., 0., 0., 1.), atol=self.config.homogeneous_atol, rtol=0.):
            reasons.append("INVALID_HOMOGENEOUS_ROW")
        rotation = matrix[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=self.config.rotation_atol, rtol=0.)
                or not np.isclose(np.linalg.det(rotation), 1., atol=self.config.rotation_atol, rtol=0.)):
            reasons.append("INVALID_SO3_ROTATION")
        return reasons

    @staticmethod
    def _project_line(points, depth, calibration, transform):
        points = np.asarray(points, dtype=float)
        depth = np.asarray(depth, dtype=float)
        u, v = points[:, 0], points[:, 1]
        if np.any(u < 0) or np.any(v < 0) or np.any(u > depth.shape[1] - 1) or np.any(v > depth.shape[0] - 1):
            return None, ("PIXEL_OUT_OF_DEPTH_BOUNDS",)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        u1, v1 = np.minimum(u0 + 1, depth.shape[1] - 1), np.minimum(v0 + 1, depth.shape[0] - 1)
        values = np.array((depth[v0, u0], depth[v0, u1], depth[v1, u0], depth[v1, u1]))
        if not np.isfinite(values).all() or (values <= 0).any():
            return None, ("INVALID_OR_MISSING_DEPTH",)
        a, b = u-u0, v-v0
        z = values[0]*(1-a)*(1-b) + values[1]*a*(1-b) + values[2]*(1-a)*b + values[3]*a*b
        camera = np.column_stack(((u-calibration.cx)*z/calibration.fx,
                                  (v-calibration.cy)*z/calibration.fy,
                                  z, np.ones(len(z))))
        return (camera @ transform.T)[:, :3], ()
