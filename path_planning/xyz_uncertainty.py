"""Synthetic-only Jacobian covariance propagation for Stage 6D.

Rotation perturbations use the left/reference-frame convention:
``T' = Exp([dt, dtheta]) T`` and ``dp_ref = dt + dtheta x p_ref``.
No result from this module is calibrated sensor uncertainty.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .calibration_evidence import CalibrationProjectionResult, CalibrationTrust
from .experimental_metric_projection import MetricGeometryStatus, MetricUncertaintyStatus


class UncertaintyValidity(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    DECLARED = "DECLARED"
    INVALID = "INVALID"


class NumericalPropagationStatus(str, Enum):
    MODELLED_SYNTHETIC = "MODELLED_SYNTHETIC"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


def _ro(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _skew(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def pinhole_jacobian(u, v, depth, fx, fy, cx, cy):
    """Jacobian w.r.t. [u, v, Z, fx, fy, cx, cy]."""
    return np.array([
        [depth/fx, 0., (u-cx)/fx, -(u-cx)*depth/(fx*fx), 0., -depth/fx, 0.],
        [0., depth/fy, (v-cy)/fy, 0., -(v-cy)*depth/(fy*fy), 0., -depth/fy],
        [0., 0., 1., 0., 0., 0., 0.],
    ])


def extrinsic_jacobian(point_reference):
    """Jacobian w.r.t. [tx, ty, tz, rx, ry, rz] for left perturbations."""
    return np.column_stack((np.eye(3), -_skew(point_reference)))


@dataclass(frozen=True)
class UncertaintyEvidence:
    source_id: str
    camera_frame: str
    reference_frame: str
    units: str
    observation_timestamp: float
    clock_domain: str
    uncertainty_source: str
    uncertainty_method: str
    uncertainty_version: str
    calibration_evidence_reference: str
    pixel_covariance: np.ndarray                 # [u, v], 2x2
    depth_sample_covariance: np.ndarray          # 4x4 or N x 4 x 4, ordered [z00,z10,z01,z11]
    intrinsics_covariance: np.ndarray             # [fx, fy, cx, cy], 4x4
    extrinsics_covariance: np.ndarray             # [tx,ty,tz,rx,ry,rz], 6x6
    zero_cross_covariance_confirmed: bool
    uncertainty_validity: UncertaintyValidity

    def __post_init__(self):
        for name in ("pixel_covariance", "depth_sample_covariance", "intrinsics_covariance", "extrinsics_covariance"):
            object.__setattr__(self, name, _ro(getattr(self, name)))


@dataclass(frozen=True)
class UncertainMetricGeometrySegment:
    name: str
    xyz: np.ndarray
    covariance_xyz: np.ndarray
    source_id: str
    source_indices: tuple[int, int] | None
    status: MetricGeometryStatus
    uncertainty_status: MetricUncertaintyStatus
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "xyz", _ro(self.xyz))
        object.__setattr__(self, "covariance_xyz", _ro(self.covariance_xyz))


@dataclass(frozen=True)
class XYZUncertaintyResult:
    segments: tuple[UncertainMetricGeometrySegment, ...]
    source_id: str | None
    camera_frame: str | None
    reference_frame: str | None
    units: str
    timestamp: float | None
    clock_domain: str | None
    transform_evidence: object | None
    calibration_evidence: object | None
    uncertainty_evidence: UncertaintyEvidence | None
    calibration_trust: CalibrationTrust | None
    propagation_status: NumericalPropagationStatus
    uncertainty_status: MetricUncertaintyStatus
    diagnostics: tuple[str, ...] = ()
    experimental: bool = True
    execution_authorized: bool = False


class ExperimentalXYZUncertaintyPropagator:
    """Propagate declared synthetic covariance after Stage 6B/6C admission."""

    def propagate(self, projection: CalibrationProjectionResult, vision, depth, calibration, uncertainty):
        diagnostics = self._validate(projection, vision, depth, calibration, uncertainty)
        if diagnostics:
            return self._result(projection, uncertainty, NumericalPropagationStatus.INVALID, tuple(diagnostics))
        if uncertainty.uncertainty_validity is not UncertaintyValidity.SYNTHETIC:
            return self._result(projection, uncertainty, NumericalPropagationStatus.UNKNOWN,
                                ("UNCERTAINTY_NOT_SYNTHETIC_MODEL",))

        metric = projection.metric_result
        segments = []
        for source_segment, line in zip(metric.segments, (vision.left, vision.right)):
            covariances = []
            for index, (point, reference_point) in enumerate(zip(line.points, source_segment.xyz)):
                weights, sample_cov = self._depth_weights_covariance(point, uncertainty.depth_sample_covariance, index)
                depth_value = self._bilinear_depth(depth, point)
                j = pinhole_jacobian(point[0], point[1], depth_value, calibration.fx, calibration.fy, calibration.cx, calibration.cy)
                camera_cov = (j[:, :2] @ uncertainty.pixel_covariance @ j[:, :2].T
                              + np.outer(j[:, 2], j[:, 2]) * float(weights @ sample_cov @ weights)
                              + j[:, 3:] @ uncertainty.intrinsics_covariance @ j[:, 3:].T)
                rotation = projection.metric_result.transform_evidence.transform_matrix[:3, :3]
                e = extrinsic_jacobian(reference_point)
                reference_cov = rotation @ camera_cov @ rotation.T + e @ uncertainty.extrinsics_covariance @ e.T
                covariances.append((reference_cov + reference_cov.T) / 2.)
            segments.append(UncertainMetricGeometrySegment(
                source_segment.name, source_segment.xyz, np.asarray(covariances), source_segment.source_id,
                source_segment.source_indices, MetricGeometryStatus.UNCERTAIN,
                MetricUncertaintyStatus.UNKNOWN,
                ("MODELLED_SYNTHETIC_COVARIANCE", "CALIBRATED_UNCERTAINTY_UNAVAILABLE"),
            ))
        return XYZUncertaintyResult(tuple(segments), metric.source_id, metric.camera_frame, metric.reference_frame,
            "m", metric.timestamp, metric.clock_domain, metric.transform_evidence, metric.calibration_evidence,
            uncertainty, projection.calibration.trust, NumericalPropagationStatus.MODELLED_SYNTHETIC,
            MetricUncertaintyStatus.UNKNOWN,
            ("MODELLED_SYNTHETIC_COVARIANCE", "CALIBRATED_UNCERTAINTY_UNAVAILABLE"))

    def _result(self, projection, uncertainty, status, diagnostics):
        metric = projection.metric_result if isinstance(projection, CalibrationProjectionResult) else None
        return XYZUncertaintyResult((), metric.source_id if metric else None, metric.camera_frame if metric else None,
            metric.reference_frame if metric else None, "m", metric.timestamp if metric else None,
            metric.clock_domain if metric else None, metric.transform_evidence if metric else None,
            metric.calibration_evidence if metric else None, uncertainty if isinstance(uncertainty, UncertaintyEvidence) else None,
            projection.calibration.trust if isinstance(projection, CalibrationProjectionResult) else None,
            status, MetricUncertaintyStatus.UNKNOWN, tuple(sorted(set(diagnostics))))

    def _validate(self, projection, vision, depth, calibration, evidence):
        reasons = []
        if not isinstance(projection, CalibrationProjectionResult) or projection.metric_result is None:
            return ("MISSING_ADMITTED_METRIC_PROJECTION",)
        metric = projection.metric_result
        if metric.status is not MetricGeometryStatus.UNCERTAIN:
            reasons.append("METRIC_PROJECTION_STATUS_INVALID")
        if not isinstance(evidence, UncertaintyEvidence):
            return tuple(reasons + ["MISSING_UNCERTAINTY_EVIDENCE"])
        checks = (
            (evidence.source_id, metric.source_id, "SOURCE_ID_MISMATCH"),
            (evidence.camera_frame, metric.camera_frame, "CAMERA_FRAME_MISMATCH"),
            (evidence.reference_frame, metric.reference_frame, "REFERENCE_FRAME_MISMATCH"),
            (evidence.units, "m", "UNITS_MISMATCH"),
            (evidence.observation_timestamp, metric.timestamp, "TIMESTAMP_MISMATCH"),
            (evidence.clock_domain, metric.clock_domain, "CLOCK_DOMAIN_MISMATCH"),
        )
        reasons.extend(code for left, right, code in checks if left != right)
        expected_reference = f"{projection.calibration.evidence.calibration_id}@{projection.calibration.evidence.calibration_version}"
        if evidence.calibration_evidence_reference != expected_reference:
            reasons.append("CALIBRATION_EVIDENCE_REFERENCE_MISMATCH")
        if not evidence.zero_cross_covariance_confirmed:
            reasons.append("CROSS_COVARIANCE_UNSPECIFIED")
        for matrix, shape, name in ((evidence.pixel_covariance, (2, 2), "PIXEL"),
                                    (evidence.intrinsics_covariance, (4, 4), "INTRINSICS"),
                                    (evidence.extrinsics_covariance, (6, 6), "EXTRINSICS")):
            reasons.extend(self._covariance_reasons(matrix, shape, name))
        depth_cov = evidence.depth_sample_covariance
        if depth_cov.ndim == 2:
            reasons.extend(self._covariance_reasons(depth_cov, (4, 4), "DEPTH"))
        elif (depth_cov.ndim == 3 and len(vision.left.points) == len(vision.right.points)
              and depth_cov.shape[0] == len(vision.left.points)):
            for item in depth_cov: reasons.extend(self._covariance_reasons(item, (4, 4), "DEPTH"))
        else:
            reasons.append("DEPTH_COVARIANCE_SHAPE")
        if np.asarray(depth).ndim != 2:
            reasons.append("INVALID_DEPTH_IMAGE")
        return tuple(sorted(set(reasons)))

    @staticmethod
    def _covariance_reasons(matrix, shape, name):
        if matrix.shape != shape: return (f"{name}_COVARIANCE_SHAPE",)
        if not np.isfinite(matrix).all(): return (f"{name}_COVARIANCE_NONFINITE",)
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.): return (f"{name}_COVARIANCE_NONSYMMETRIC",)
        if np.linalg.eigvalsh((matrix+matrix.T)/2.).min() < -1e-10: return (f"{name}_COVARIANCE_NOT_PSD",)
        return ()

    @staticmethod
    def _depth_weights_covariance(point, covariance, index):
        u, v = point; a, b = u-np.floor(u), v-np.floor(v)
        weights = np.array([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b])
        return weights, covariance if covariance.ndim == 2 else covariance[index]

    @staticmethod
    def _bilinear_depth(depth, point):
        u, v = point; u0, v0 = int(np.floor(u)), int(np.floor(v))
        u1, v1 = min(u0+1, depth.shape[1]-1), min(v0+1, depth.shape[0]-1)
        a, b = u-u0, v-v0
        values = np.array([depth[v0,u0], depth[v0,u1], depth[v1,u0], depth[v1,u1]], float)
        return float(np.array([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b]) @ values)
