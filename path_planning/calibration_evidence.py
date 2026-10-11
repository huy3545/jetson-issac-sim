"""Opt-in calibration evidence trust layer for Stage 6B metric projection.

Evidence validation establishes metadata/artifact integrity only.  It does not
assert physical calibration accuracy and cannot authorize execution.
"""
from dataclasses import dataclass
from enum import Enum
import hashlib
import re

from .experimental_metric_projection import (
    ExperimentalVisionMetricProjectionAdapter,
    MetricGeometryResult,
    MetricGeometryStatus,
    MetricUncertaintyStatus,
    TransformEvidence,
    TransformValidity,
)


class CalibrationTrust(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    DECLARED = "DECLARED"
    EVIDENCE_VALIDATED = "EVIDENCE_VALIDATED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CalibrationEvidence:
    calibration_id: str
    calibration_version: str
    camera_id: str
    camera_frame: str
    reference_frame: str
    intrinsics_id: str
    intrinsics_version: str
    extrinsics_id: str
    extrinsics_version: str
    calibration_source: str
    calibration_method: str
    calibration_timestamp: float
    clock_domain: str
    units: str
    source_id: str
    calibration_artifact_reference: str | None = None
    calibration_artifact_digest: str | None = None
    validation_method: str | None = None
    validation_timestamp: float | None = None
    validation_result: bool | None = None
    validation_diagnostics: tuple[str, ...] = ()
    declared_trust: CalibrationTrust = CalibrationTrust.DECLARED


@dataclass(frozen=True)
class CalibrationValidationResult:
    trust: CalibrationTrust
    diagnostics: tuple[str, ...]
    evidence: CalibrationEvidence | None


@dataclass(frozen=True)
class CalibrationProjectionResult:
    calibration: CalibrationValidationResult
    metric_result: MetricGeometryResult | None
    experimental: bool = True
    execution_authorized: bool = False


class CalibrationEvidenceValidator:
    """Validate supplied evidence without reading artifacts or inventing proof."""

    _DIGEST = re.compile(r"^[0-9a-f]{64}$")

    def validate(self, evidence, artifact_bytes=None):
        if not isinstance(evidence, CalibrationEvidence):
            return CalibrationValidationResult(CalibrationTrust.INVALID, ("MISSING_CALIBRATION_EVIDENCE",), None)
        required = (
            evidence.calibration_id, evidence.calibration_version, evidence.camera_id,
            evidence.camera_frame, evidence.reference_frame, evidence.intrinsics_id,
            evidence.intrinsics_version, evidence.extrinsics_id, evidence.extrinsics_version,
            evidence.calibration_source, evidence.calibration_method, evidence.clock_domain,
            evidence.units, evidence.source_id,
        )
        reasons = list(evidence.validation_diagnostics)
        if not all(required) or evidence.units != "m":
            reasons.append("MISSING_OR_INVALID_CALIBRATION_METADATA")
        if not self._finite(evidence.calibration_timestamp):
            reasons.append("INVALID_CALIBRATION_TIMESTAMP")
        has_reference = evidence.calibration_artifact_reference is not None
        has_digest = evidence.calibration_artifact_digest is not None
        if has_reference != has_digest:
            reasons.append("INCOMPLETE_ARTIFACT_REFERENCE")
        if has_digest and not self._DIGEST.fullmatch(evidence.calibration_artifact_digest):
            reasons.append("INVALID_ARTIFACT_DIGEST_FORMAT")
        if artifact_bytes is not None:
            if not has_digest:
                reasons.append("ARTIFACT_DIGEST_MISSING")
            elif hashlib.sha256(artifact_bytes).hexdigest() != evidence.calibration_artifact_digest:
                reasons.append("ARTIFACT_DIGEST_MISMATCH")
        if evidence.validation_result is True and (not evidence.validation_method or not self._finite(evidence.validation_timestamp)):
            reasons.append("INVALID_VALIDATION_CLAIM")
        if evidence.validation_result is False:
            reasons.append("VALIDATION_REPORTED_FAILURE")
        if reasons:
            return CalibrationValidationResult(CalibrationTrust.INVALID, tuple(sorted(set(reasons))), evidence)
        if evidence.declared_trust is CalibrationTrust.SYNTHETIC:
            return CalibrationValidationResult(CalibrationTrust.SYNTHETIC, (), evidence)
        # Only bytes supplied to this call permit an integrity check.  Caller
        # metadata, including a self-declared EVIDENCE_VALIDATED label, is DECLARED.
        if (artifact_bytes is not None and has_digest and evidence.validation_result is True
                and evidence.validation_method and self._finite(evidence.validation_timestamp)):
            return CalibrationValidationResult(CalibrationTrust.EVIDENCE_VALIDATED, (), evidence)
        return CalibrationValidationResult(CalibrationTrust.DECLARED, (), evidence)

    @staticmethod
    def _finite(value):
        try:
            return bool(float(value) == float(value) and abs(float(value)) != float("inf"))
        except (TypeError, ValueError):
            return False


class CalibrationTrustedProjectionAdapter:
    """Evidence-first wrapper retaining Stage 6B's conservative output policy."""

    def __init__(self, projection_adapter=None, validator=None):
        self._projection = projection_adapter or ExperimentalVisionMetricProjectionAdapter()
        self._validator = validator or CalibrationEvidenceValidator()

    def project(self, vision, depth, camera_calibration, transform_evidence: TransformEvidence,
                calibration_evidence, artifact_bytes=None):
        validated = self._validator.validate(calibration_evidence, artifact_bytes)
        consistency = self._consistency_diagnostics(validated.evidence, transform_evidence)
        if validated.trust is CalibrationTrust.INVALID or consistency:
            diagnostics = tuple(sorted(set(validated.diagnostics + tuple(consistency))))
            invalid = CalibrationValidationResult(CalibrationTrust.INVALID, diagnostics, validated.evidence)
            return CalibrationProjectionResult(invalid, None)
        if (validated.trust is CalibrationTrust.SYNTHETIC
                and transform_evidence.transform_validity is not TransformValidity.SYNTHETIC):
            invalid = CalibrationValidationResult(CalibrationTrust.INVALID, ("SYNTHETIC_TRUST_MISMATCH",), validated.evidence)
            return CalibrationProjectionResult(invalid, None)
        return CalibrationProjectionResult(validated, self._projection.project(
            vision, depth, camera_calibration, transform_evidence,
        ))

    @staticmethod
    def _consistency_diagnostics(calibration, transform):
        if not isinstance(transform, TransformEvidence):
            return ("MISSING_TRANSFORM_EVIDENCE",)
        if calibration is None:
            return ("MISSING_CALIBRATION_EVIDENCE",)
        expected = (
            (calibration.camera_frame, transform.camera_frame, "CAMERA_FRAME_MISMATCH"),
            (calibration.reference_frame, transform.reference_frame, "REFERENCE_FRAME_MISMATCH"),
            (calibration.units, transform.units, "UNITS_MISMATCH"),
            (calibration.intrinsics_id, transform.intrinsics_id, "INTRINSICS_ID_MISMATCH"),
            (calibration.intrinsics_version, transform.intrinsics_version, "INTRINSICS_VERSION_MISMATCH"),
            (calibration.extrinsics_id, transform.extrinsics_id, "EXTRINSICS_ID_MISMATCH"),
            (calibration.extrinsics_version, transform.extrinsics_version, "EXTRINSICS_VERSION_MISMATCH"),
            (calibration.calibration_source, transform.calibration_source, "CALIBRATION_SOURCE_MISMATCH"),
            (calibration.source_id, transform.source_id, "SOURCE_ID_MISMATCH"),
            (calibration.clock_domain, transform.clock_domain, "CLOCK_DOMAIN_MISMATCH"),
        )
        return tuple(code for left, right, code in expected if left != right)
