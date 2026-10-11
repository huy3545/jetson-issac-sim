"""Bounded JSON ingestion for caller-supplied Stage 6D uncertainty evidence."""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import numpy as np

from .xyz_uncertainty import UncertaintyEvidence, UncertaintyValidity, ExperimentalXYZUncertaintyPropagator


class ArtifactTrust(str, Enum):
    DECLARED = "DECLARED"
    EVIDENCE_VALIDATED = "EVIDENCE_VALIDATED"
    INVALID = "INVALID"


class CrossCovariancePolicy(str, Enum):
    EXPLICIT_ZERO_ASSUMPTION = "EXPLICIT_ZERO_ASSUMPTION"
    PROVIDED_CROSS_COVARIANCE = "PROVIDED_CROSS_COVARIANCE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class UncertaintyArtifactResult:
    trust: ArtifactTrust
    artifact_id: str | None
    artifact_version: str | None
    schema_version: str | None
    provenance: tuple[tuple[str, object], ...]
    cross_covariance_policy: CrossCovariancePolicy | None
    correlation_metadata: tuple[tuple[str, object], ...]
    diagnostics: tuple[str, ...]
    uncertainty_evidence: UncertaintyEvidence | None
    experimental: bool = True
    execution_authorized: bool = False


class UncertaintyArtifactIngestor:
    SCHEMA_VERSION = "1.0"
    MAX_BYTES = 1_000_000
    MAX_NESTING = 16
    ORDERING = {"pixel": ["u", "v"], "depth": ["z00", "z10", "z01", "z11"],
                "intrinsics": ["fx", "fy", "cx", "cy"],
                "extrinsics": ["tx", "ty", "tz", "rx", "ry", "rz"]}
    SHAPES = {"pixel": (2, 2), "depth": (4, 4), "intrinsics": (4, 4), "extrinsics": (6, 6)}

    def __init__(self, max_bytes=MAX_BYTES, max_nesting=MAX_NESTING):
        self.max_bytes, self.max_nesting = max_bytes, max_nesting

    def ingest(self, artifact_bytes, expected_digest=None, *, calibration_evidence_reference=None,
               source_id=None, camera_frame=None, reference_frame=None, observation_timestamp=None,
               clock_domain=None):
        if not isinstance(artifact_bytes, (bytes, bytearray)):
            return self._invalid(("ARTIFACT_BYTES_REQUIRED",))
        if len(artifact_bytes) > self.max_bytes:
            return self._invalid(("ARTIFACT_TOO_LARGE",))
        digest_ok = expected_digest is not None and hashlib.sha256(artifact_bytes).hexdigest() == expected_digest
        if expected_digest is not None and not digest_ok:
            return self._invalid(("ARTIFACT_DIGEST_MISMATCH",))
        try:
            document = json.loads(bytes(artifact_bytes).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._invalid(("MALFORMED_JSON",))
        if self._depth(document) > self.max_nesting:
            return self._invalid(("JSON_NESTING_LIMIT",))
        if not isinstance(document, dict):
            return self._invalid(("ARTIFACT_ROOT_NOT_OBJECT",))
        reasons = self._validate_document(document, calibration_evidence_reference, source_id, camera_frame,
                                          reference_frame, observation_timestamp, clock_domain)
        policy = self._policy(document.get("cross_covariance_policy"))
        identity = self._identity(document)
        if reasons:
            return UncertaintyArtifactResult(ArtifactTrust.INVALID, *identity, policy,
                self._freeze_mapping(document.get("correlation_metadata")), tuple(sorted(set(reasons))), None)
        if policy is CrossCovariancePolicy.UNKNOWN:
            return UncertaintyArtifactResult(self._trust(digest_ok), *identity, policy,
                self._freeze_mapping(document.get("correlation_metadata")), ("CROSS_COVARIANCE_UNKNOWN",), None)
        if policy is CrossCovariancePolicy.PROVIDED_CROSS_COVARIANCE:
            return UncertaintyArtifactResult(self._trust(digest_ok), *identity, policy,
                self._freeze_mapping(document.get("correlation_metadata")), ("UNSUPPORTED_FOR_PROPAGATION",), None)
        evidence = self._to_evidence(document)
        return UncertaintyArtifactResult(self._trust(digest_ok), *identity, policy,
            self._freeze_mapping(document.get("correlation_metadata")), (), evidence)

    def _validate_document(self, d, expected_cal, source, camera, reference, timestamp, domain):
        required = ("schema_version", "artifact_id", "artifact_version", "source_id", "camera_frame",
                    "reference_frame", "units", "observation_timestamp", "clock_domain", "uncertainty_source",
                    "uncertainty_method", "uncertainty_version", "calibration_evidence_reference",
                    "variable_ordering", "covariance_blocks", "cross_covariance_policy", "correlation_metadata",
                    "validation_metadata")
        reasons = ["MISSING_"+key.upper() for key in required if key not in d or d[key] in (None, "")]
        if d.get("schema_version") != self.SCHEMA_VERSION: reasons.append("UNSUPPORTED_SCHEMA_VERSION")
        if d.get("units") != "m": reasons.append("UNITS_MISMATCH")
        for key, expected, code in (("source_id",source,"SOURCE_ID_MISMATCH"),("camera_frame",camera,"CAMERA_FRAME_MISMATCH"),
                                    ("reference_frame",reference,"REFERENCE_FRAME_MISMATCH"),("observation_timestamp",timestamp,"TIMESTAMP_MISMATCH"),
                                    ("clock_domain",domain,"CLOCK_DOMAIN_MISMATCH"),("calibration_evidence_reference",expected_cal,"CALIBRATION_REFERENCE_MISMATCH")):
            if expected is not None and d.get(key) != expected: reasons.append(code)
        if d.get("variable_ordering") != self.ORDERING: reasons.append("INVALID_VARIABLE_ORDERING")
        blocks = d.get("covariance_blocks")
        if not isinstance(blocks, dict): return reasons + ["INVALID_COVARIANCE_BLOCKS"]
        for name, shape in self.SHAPES.items():
            if name not in blocks: reasons.append("MISSING_"+name.upper()+"_COVARIANCE")
            else: reasons.extend(self._covariance_reasons(blocks[name], shape, name.upper()))
        policy = self._policy(d.get("cross_covariance_policy"))
        if policy is None: reasons.append("INVALID_CROSS_COVARIANCE_POLICY")
        if policy is CrossCovariancePolicy.EXPLICIT_ZERO_ASSUMPTION:
            metadata=d.get("correlation_metadata")
            if not isinstance(metadata,dict) or not metadata.get("zero_cross_covariance_assumption") or not metadata.get("assumption_source"):
                reasons.append("ZERO_ASSUMPTION_PROVENANCE_MISSING")
        return reasons

    @staticmethod
    def _covariance_reasons(raw, shape, name):
        try: matrix=np.asarray(raw,dtype=float)
        except (TypeError,ValueError): return (f"{name}_COVARIANCE_INVALID",)
        if matrix.shape != shape: return (f"{name}_COVARIANCE_SHAPE",)
        if not np.isfinite(matrix).all(): return (f"{name}_COVARIANCE_NONFINITE",)
        if not np.allclose(matrix,matrix.T,atol=1e-12,rtol=0.): return (f"{name}_COVARIANCE_NONSYMMETRIC",)
        if np.linalg.eigvalsh((matrix+matrix.T)/2).min() < -1e-10: return (f"{name}_COVARIANCE_NOT_PSD",)
        return ()

    @staticmethod
    def _depth(value):
        if isinstance(value,dict): return 1+max((UncertaintyArtifactIngestor._depth(v) for v in value.values()),default=0)
        if isinstance(value,list): return 1+max((UncertaintyArtifactIngestor._depth(v) for v in value),default=0)
        return 0
    @staticmethod
    def _policy(value):
        try: return CrossCovariancePolicy(value)
        except ValueError: return None
    @staticmethod
    def _trust(digest_ok): return ArtifactTrust.EVIDENCE_VALIDATED if digest_ok else ArtifactTrust.DECLARED
    @staticmethod
    def _freeze_mapping(value):
        return tuple(sorted(value.items())) if isinstance(value,dict) else ()
    @staticmethod
    def _identity(d): return (d.get("artifact_id"),d.get("artifact_version"),d.get("schema_version"),
                              tuple(sorted((key,d.get(key)) for key in ("source_id","camera_frame","reference_frame","units","observation_timestamp","clock_domain","calibration_evidence_reference"))))
    def _invalid(self,reasons): return UncertaintyArtifactResult(ArtifactTrust.INVALID,None,None,None,(),None,(),tuple(reasons),None)
    @staticmethod
    def _to_evidence(d):
        blocks=d["covariance_blocks"]; metadata=d["validation_metadata"]
        validity=UncertaintyValidity.SYNTHETIC if metadata.get("evidence_kind")=="SYNTHETIC" else UncertaintyValidity.DECLARED
        return UncertaintyEvidence(d["source_id"],d["camera_frame"],d["reference_frame"],d["units"],d["observation_timestamp"],d["clock_domain"],d["uncertainty_source"],d["uncertainty_method"],d["uncertainty_version"],d["calibration_evidence_reference"],blocks["pixel"],blocks["depth"],blocks["intrinsics"],blocks["extrinsics"],True,validity)
