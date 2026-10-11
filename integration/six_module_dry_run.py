"""Stage 7D-1A passive six-module trace with evidence-preserving contracts.

The harness invokes the public Vision-to-Path adapter, then records passive
Motion, Safety, Control and Logging boundaries. It never creates targets or
imports a machine controller. A sink acknowledgement is weaker than durable
persistence.
"""
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import VisionResult

SCHEMA_VERSION = "1.1"


class DryRunStatus(str, Enum):
    ACCEPTED = "ACCEPTED"; REJECTED = "REJECTED"; BLOCKED = "BLOCKED"; ERROR = "ERROR"; SKIPPED = "SKIPPED"


class AuditWriteStatus(str, Enum):
    ZERO_ACKNOWLEDGED = "ZERO_ACKNOWLEDGED"
    PARTIAL_ACKNOWLEDGED = "PARTIAL_ACKNOWLEDGED"
    COMPLETE_ACKNOWLEDGED = "COMPLETE_ACKNOWLEDGED"


@dataclass(frozen=True)
class DryRunStageRecord:
    schema_version: str; run_id: str; correlation_id: str; stage_id: str; status: DryRunStatus
    source_frame_id: str | int | None; timestamp: float | None; clock_domain: str | None
    coordinate_frame: str | None; units: str | None; config_version: str | None
    source_path_id: str | None; source_segment_ids: tuple[str, ...]
    calibration_id: str | None; transform_id: str | None; provenance: tuple[str, ...]
    confidence: float | None; uncertainty_status: str | None; geometry_status: str | None
    admission_status: str; reason_codes: tuple[str, ...]; diagnostics: tuple[str, ...] = ()
    calibration_evidence_status: str = "MISSING"; transform_evidence_status: str = "MISSING"
    uncertainty_evidence_status: str = "UNKNOWN"; execution_authorized: bool = False

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION: raise ValueError("unsupported schema")
        if not self.run_id or not self.correlation_id or not self.stage_id: raise ValueError("missing trace identity")
        if self.execution_authorized: raise ValueError("dry run cannot authorize execution")


@dataclass(frozen=True)
class SixModuleDryRunResult:
    schema_version: str; run_id: str; correlation_id: str; records: tuple[DryRunStageRecord, ...]
    final_status: DryRunStatus; audit_complete: bool; audit_write_status: AuditWriteStatus
    records_attempted: int; records_confirmed: int; pipeline_final_status: DryRunStatus
    execution_authorized: bool = False

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION or len(self.records) != 6: raise ValueError("six records are required")
        if self.final_status is not self.pipeline_final_status: raise ValueError("pipeline status mismatch")
        if self.execution_authorized or any(record.execution_authorized for record in self.records): raise ValueError("execution authorization forbidden")
        if any(record.run_id != self.run_id or record.correlation_id != self.correlation_id for record in self.records): raise ValueError("trace identity mismatch")
        if self.audit_complete != (self.records_confirmed == 6): raise ValueError("audit completion does not match acknowledgements")


class InMemorySixModuleAuditSink:
    """Successful ``emit`` is acknowledgement only; it is not durable persistence."""
    def __init__(self): self.records = []
    def emit(self, record): self.records.append(record)


class SixModuleDryRunHarness:
    """Trace Vision→Path→Motion→Safety→Control simulation→Audit fail-closed."""
    ORDER = ("vision", "path", "motion", "safety", "machine_control_simulation", "data_logging")
    def __init__(self, audit_sink=None):
        self.audit_sink = audit_sink or InMemorySixModuleAuditSink()
        self.path_adapter = VisionPathAdapter(solver=CornerAwareTwoLineSolver(), require_point_provenance=True)

    def run(self, vision, *, run_id=None, correlation_id=None, calibration_id=None, transform_id=None, uncertainty_status="UNKNOWN", expected_frame_id=None, expected_timestamp=None, expected_clock_domain=None, config_version="stage7d1-synthetic"):
        run_id = run_id or uuid4().hex; correlation_id = correlation_id or run_id
        if not isinstance(vision, VisionResult):
            return self._finish([self._record(run_id, correlation_id, "vision", DryRunStatus.ERROR, None, ("INVALID_VISION_RESULT",), config_version=config_version)], run_id, correlation_id)
        identity, metadata_reasons = self._identity(calibration_id, transform_id, uncertainty_status, config_version)
        confidence, provenance = self._confidence(vision), self._provenance(vision)
        try:
            preflight_reasons = self.path_adapter.validate_observation(vision)
        except (TypeError, ValueError, AttributeError) as exc:
            return self._finish([self._record(run_id, correlation_id, "vision", DryRunStatus.ERROR, vision, ("VISION_ADAPTER_ERROR",), confidence=confidence, provenance=provenance, diagnostics=(type(exc).__name__,), **identity)], run_id, correlation_id, vision=vision, identity=identity, confidence=confidence, provenance=provenance)
        vision_reasons = self._vision_reasons(vision, preflight_reasons, expected_frame_id, expected_timestamp, expected_clock_domain) + tuple(metadata_reasons)
        vision_status = DryRunStatus.ACCEPTED if not vision_reasons else DryRunStatus.REJECTED
        records = [self._record(run_id, correlation_id, "vision", vision_status, vision, self._unique(vision_reasons), confidence=confidence, provenance=provenance, geometry_status="OBSERVED" if vision_status is DryRunStatus.ACCEPTED else "UNTRUSTED", diagnostics=preflight_reasons, **identity)]
        if vision_status is not DryRunStatus.ACCEPTED:
            path_record = self._record(run_id, correlation_id, "path", DryRunStatus.SKIPPED, vision, ("VISION_DEPENDENCY_REJECTED", records[0].reason_codes[0]), confidence=confidence, provenance=provenance, geometry_status="UNAVAILABLE", diagnostics=preflight_reasons, **identity)
        else:
            try:
                path_result = self.path_adapter.plan(vision)
            except (TypeError, ValueError, AttributeError) as exc:
                path_result = None
                path_record = self._record(run_id, correlation_id, "path", DryRunStatus.ERROR, vision, ("PATH_ADAPTER_ERROR",), confidence=confidence, provenance=provenance, geometry_status="ERROR", diagnostics=(type(exc).__name__,), **identity)
        if vision_status is DryRunStatus.ACCEPTED and path_result is not None and path_result.status is not VisionPathStatus.VERIFIED:
            path_record = self._record(run_id, correlation_id, "path", DryRunStatus.REJECTED, vision, path_result.reason_codes or ("PATH_GEOMETRY_REJECTED",), confidence=confidence, provenance=provenance, geometry_status="REJECTED", diagnostics=self._solver_diagnostics(path_result), **identity)
        elif vision_status is DryRunStatus.ACCEPTED and path_result is not None:
            # VisionPathResult publishes no trustworthy path/segment ID; leave those fields unavailable.
            path_record = self._record(run_id, correlation_id, "path", DryRunStatus.BLOCKED, vision, ("STRICT_GATE_B_BLOCKED", "CALIBRATION_UNVERIFIED", "TRANSFORM_UNVERIFIED", "XYZ_UNCERTAINTY_UNKNOWN"), confidence=confidence, provenance=provenance, geometry_status="SYNTHETIC_GEOMETRY_VERIFIED", diagnostics=self._solver_diagnostics(path_result), **identity)
        records.append(path_record)
        records.append(self._dependency_record(run_id, correlation_id, "motion", vision, path_record, confidence, provenance, identity))
        records.append(self._dependency_record(run_id, correlation_id, "safety", vision, records[-1], confidence, provenance, identity))
        records.append(self._dependency_record(run_id, correlation_id, "machine_control_simulation", vision, records[-1], confidence, provenance, identity))
        return self._finish(records, run_id, correlation_id, vision=vision, identity=identity, confidence=confidence, provenance=provenance, path_record=path_record)

    def _dependency_record(self, run, correlation, stage, vision, upstream, confidence, provenance, identity):
        if upstream.status is DryRunStatus.BLOCKED: status, reasons = DryRunStatus.BLOCKED, ("UPSTREAM_STRICT_ADMISSION_BLOCKED", upstream.reason_codes[0])
        elif upstream.status is DryRunStatus.REJECTED: status, reasons = DryRunStatus.SKIPPED, ("UPSTREAM_REJECTED", upstream.reason_codes[0])
        elif upstream.status is DryRunStatus.SKIPPED: status, reasons = DryRunStatus.SKIPPED, ("UPSTREAM_SKIPPED", upstream.reason_codes[0])
        else: status, reasons = DryRunStatus.SKIPPED, ("UPSTREAM_ERROR", upstream.reason_codes[0])
        if stage == "machine_control_simulation": reasons += ("PASSIVE_SIMULATION_ONLY",)
        return self._record(run, correlation, stage, status, vision, reasons, confidence=confidence, provenance=provenance, geometry_status=upstream.geometry_status, source_path_id=upstream.source_path_id, source_segment_ids=upstream.source_segment_ids, diagnostics=upstream.diagnostics, **identity)

    def _finish(self, records, run_id, correlation_id, *, vision=None, identity=None, confidence=None, provenance=(), path_record=None):
        identity = identity or self._identity(None, None, "UNKNOWN", "stage7d1-synthetic")[0]
        # Validation failures still produce the complete six-stage trace.  These are
        # explicit skipped dependencies, never fabricated executions.
        while len(records) < 5:
            upstream = records[-1]
            records.append(self._dependency_record(run_id, correlation_id, self.ORDER[len(records)], vision,
                                                   upstream, confidence, provenance, identity))
        path_record = path_record or records[-1]
        logging = self._record(run_id, correlation_id, "data_logging", DryRunStatus.ACCEPTED, vision, (), confidence=confidence, provenance=provenance, geometry_status=path_record.geometry_status, source_path_id=path_record.source_path_id, source_segment_ids=path_record.source_segment_ids, **identity)
        attempted = confirmed = 0; audit_error = None
        for record in tuple(records) + (logging,):
            attempted += 1
            try: self.audit_sink.emit(record); confirmed += 1
            except Exception as exc: audit_error = type(exc).__name__; break
        complete = confirmed == 6
        write_status = AuditWriteStatus.COMPLETE_ACKNOWLEDGED if complete else (AuditWriteStatus.ZERO_ACKNOWLEDGED if confirmed == 0 else AuditWriteStatus.PARTIAL_ACKNOWLEDGED)
        if not complete:
            logging = self._record(run_id, correlation_id, "data_logging", DryRunStatus.ERROR, vision, ("AUDIT_LOGGING_FAILED",), confidence=confidence, provenance=provenance, geometry_status=path_record.geometry_status, source_path_id=path_record.source_path_id, source_segment_ids=path_record.source_segment_ids, diagnostics=((audit_error,) if audit_error else ()), **identity)
        all_records = tuple(records) + (logging,)
        final = DryRunStatus.ERROR if not complete else self._final_status(all_records)
        return SixModuleDryRunResult(SCHEMA_VERSION, run_id, correlation_id, all_records, final, complete, write_status, attempted, confirmed, final, False)

    @staticmethod
    def _final_status(records):
        for status in (DryRunStatus.ERROR, DryRunStatus.REJECTED, DryRunStatus.BLOCKED, DryRunStatus.SKIPPED):
            if any(record.status is status for record in records): return status
        return DryRunStatus.ACCEPTED
    @staticmethod
    def _solver_diagnostics(result): return tuple(getattr(getattr(getattr(result, "solver_result", None), "diagnostics", None), "excluded", ()) or ())
    @staticmethod
    def _unique(values): return tuple(dict.fromkeys(values))
    @staticmethod
    def _confidence(vision):
        try: return min(float(vision.left.confidence), float(vision.right.confidence))
        except (TypeError, ValueError): return None
    @staticmethod
    def _provenance(vision):
        values = []
        for name, line in (("left", vision.left), ("right", vision.right)):
            if line.point_provenance is None: values.append(f"{name}:POINT_PROVENANCE_UNAVAILABLE")
            else: values.extend(f"{name}:{point.source_index}" if point.source_index is not None else f"{name}:SOURCE_INDEX_UNAVAILABLE" for point in line.point_provenance)
        return tuple(values)
    @staticmethod
    def _identity(calibration_id, transform_id, uncertainty_status, config_version):
        reasons = []
        def declared(value, field):
            if value is None: return None, "MISSING"
            if not isinstance(value, str) or not value.strip(): reasons.append(f"INVALID_{field}"); return None, "INVALID"
            return value, "DECLARED_UNVERIFIED"
        calibration_id, calibration_status = declared(calibration_id, "CALIBRATION_ID")
        transform_id, transform_status = declared(transform_id, "TRANSFORM_ID")
        if uncertainty_status not in ("UNKNOWN", "KNOWN", "EXCEEDED"): reasons.append("INVALID_UNCERTAINTY_STATUS"); uncertainty_status, uncertainty_evidence = "UNKNOWN", "INVALID"
        else: uncertainty_evidence = "DECLARED_UNVERIFIED" if uncertainty_status == "KNOWN" else uncertainty_status
        if not isinstance(config_version, str) or not config_version: reasons.append("INVALID_CONFIG_VERSION"); config_version = None
        return ({"calibration_id": calibration_id, "transform_id": transform_id, "uncertainty_status": uncertainty_status, "config_version": config_version, "calibration_evidence_status": calibration_status, "transform_evidence_status": transform_status, "uncertainty_evidence_status": uncertainty_evidence}, tuple(reasons))
    def _vision_reasons(self, vision, preflight_reasons, expected_frame_id, expected_timestamp, expected_clock_domain):
        reasons = list(preflight_reasons)
        if expected_frame_id is not None and vision.frame_id != expected_frame_id: reasons.append("SOURCE_FRAME_ID_MISMATCH")
        if expected_timestamp is not None and vision.frame_timestamp != expected_timestamp: reasons.append("TIMESTAMP_MISMATCH")
        if expected_clock_domain is not None and vision.frame_clock_domain != expected_clock_domain: reasons.append("CLOCK_DOMAIN_MISMATCH")
        return self._unique(reasons)
    @staticmethod
    def _record(run, correlation, stage, status, vision, reasons, *, confidence=None, provenance=(), geometry_status=None, source_path_id=None, source_segment_ids=(), calibration_id=None, transform_id=None, uncertainty_status="UNKNOWN", config_version=None, diagnostics=(), calibration_evidence_status="MISSING", transform_evidence_status="MISSING", uncertainty_evidence_status="UNKNOWN"):
        return DryRunStageRecord(SCHEMA_VERSION, run, correlation, stage, status, getattr(vision, "frame_id", None), getattr(vision, "frame_timestamp", None), getattr(vision, "frame_clock_domain", None), getattr(vision, "coordinate_frame", None), getattr(vision, "units", None), config_version, source_path_id, tuple(source_segment_ids), calibration_id, transform_id, tuple(provenance), confidence, uncertainty_status, geometry_status, status.value, tuple(reasons), tuple(diagnostics), calibration_evidence_status, transform_evidence_status, uncertainty_evidence_status, False)
