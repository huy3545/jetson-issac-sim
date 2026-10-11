"""Non-executable Stage 7B coordination of existing V1.1 public adapters.

This module intentionally has no import of controller or logging_data.  Its
in-memory audit sink is passive and exists only for contract integration tests.
"""
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from config import Config
from motion_planning.experimental_adapter import ExperimentalMotionAdapter, HandoffMode
from motion_planning.segment_motion_planner import SegmentMotionPlanner
from path_planning.path_qualification import PathQualificationStatus, UncertaintyStatus
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import ObservationStatus, VisionResult

from .contracts import AdmissionStatus, IntegrationEvent, IntegrationRecord, IntegrationRunResult


class InMemoryAuditSink:
    """Passive structured sink.  It cannot authorize or modify a decision."""
    def __init__(self):
        self.events = []

    def emit(self, event: IntegrationEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class SafetyAssessment:
    record: IntegrationRecord
    execution_authorized: bool = False


class ExperimentalSafetyAssessment:
    """A fail-closed evidence review, separate from SafetyMonitor V1."""
    REQUIRED = ("source_frame_id", "timestamp", "clock_domain", "coordinate_frame", "units", "config_version")

    def assess(self, record: IntegrationRecord, *, has_gap: bool = False,
               machine_state_present: bool = False) -> SafetyAssessment:
        reasons = []
        for field in self.REQUIRED:
            if getattr(record, field) in (None, ""):
                reasons.append(f"MISSING_{field.upper()}")
        if record.units != "m":
            reasons.append("UNITS_NOT_METRIC")
        if record.timestamp is None or not np.isfinite(record.timestamp):
            reasons.append("INVALID_TIMESTAMP")
        if not machine_state_present:
            reasons.append("MACHINE_STATE_EVIDENCE_MISSING")
        if record.admission_status is not AdmissionStatus.ACCEPTED:
            reasons.append("UPSTREAM_NOT_ACCEPTED")
        if record.qualification_status != PathQualificationStatus.VERIFIED.value:
            reasons.append("QUALIFICATION_NOT_VERIFIED")
        if record.uncertainty_status != UncertaintyStatus.KNOWN.value:
            reasons.append("UNCERTAINTY_NOT_VERIFIED")
        if has_gap:
            reasons.append("GAPPED_SEGMENT")
        # All Stage 7B records remain research data even when their local checks pass.
        reasons.append("EXECUTION_NOT_AUTHORIZED")
        status = AdmissionStatus.REJECTED
        assessed = IntegrationRecord(**{**record.__dict__, "stage_id": "safety_assessment",
                                        "admission_status": status,
                                        "reason_codes": tuple(dict.fromkeys(record.reason_codes + tuple(reasons))),
                                        "execution_authorized": False})
        return SafetyAssessment(assessed)


class ExperimentalIntegrationHarness:
    """Coordinates public adapters without creating geometry or commands.

    Vision-to-planar admission and Phase-3-to-motion admission are both real
    adapter calls.  The repository has no contract proving that their results
    describe the same source run, so the bridge is explicitly BLOCKED.
    """
    def __init__(self, config: Config | None = None, audit_sink=None):
        self.config = config or Config()
        self.audit_sink = audit_sink or InMemoryAuditSink()
        self.vision_adapter = VisionPathAdapter()
        self.motion_adapter = ExperimentalMotionAdapter()
        self.segment_planner = SegmentMotionPlanner(self.config.motion)
        self.safety = ExperimentalSafetyAssessment()

    def run(self, vision: VisionResult, qualified_path=None, *, run_id: str | None = None,
            correlation_id: str | None = None) -> IntegrationRunResult:
        run_id = run_id or uuid4().hex
        correlation_id = correlation_id or run_id
        records, events, audit_complete = [], [], True

        def publish(stage, event_type, record=None, reasons=(), diagnostics=()):
            nonlocal audit_complete
            event = IntegrationEvent(run_id, correlation_id, stage, event_type,
                                     record.timestamp if record else self._timestamp(vision),
                                     tuple(reasons), tuple(diagnostics))
            events.append(event)
            try:
                self.audit_sink.emit(event)
            except Exception as exc:  # passive logging must not alter the decision
                audit_complete = False
                events.append(IntegrationEvent(run_id, correlation_id, "audit_logging", "ERROR",
                                               self._timestamp(vision), ("AUDIT_LOGGING_FAILED",), (type(exc).__name__,)))

        publish("integration", "STARTED")
        vision_record = self._vision_record(run_id, correlation_id, vision)
        records.append(vision_record); publish("vision", vision_record.admission_status.value, vision_record,
                                               vision_record.reason_codes, vision_record.diagnostics)

        path_record = self._path_record(run_id, correlation_id, vision, vision_record)
        records.append(path_record); publish("path", path_record.admission_status.value, path_record,
                                             path_record.reason_codes, path_record.diagnostics)

        bridge = self._blocked_bridge(run_id, correlation_id, vision)
        records.append(bridge); publish("vision_to_phase3", bridge.admission_status.value, bridge, bridge.reason_codes)

        motion_record = self._motion_record(run_id, correlation_id, qualified_path, vision)
        records.append(motion_record); publish("motion", motion_record.admission_status.value, motion_record,
                                               motion_record.reason_codes, motion_record.diagnostics)

        assessment = self.safety.assess(
            motion_record, has_gap=self._has_gap(qualified_path),
            machine_state_present=getattr(vision, "machine_state_at_frame", None) is not None)
        records.append(assessment.record); publish("safety_assessment", assessment.record.admission_status.value,
                                                   assessment.record, assessment.record.reason_codes)

        final = AdmissionStatus.BLOCKED if any(r.admission_status is AdmissionStatus.BLOCKED for r in records) else AdmissionStatus.REJECTED
        publish("integration", final.value, reasons=tuple(code for r in records for code in r.reason_codes))
        return IntegrationRunResult(run_id, correlation_id, tuple(records), tuple(events), final, audit_complete)

    def _vision_record(self, run, correlation, vision):
        if not isinstance(vision, VisionResult):
            return self._record(run, correlation, "vision", AdmissionStatus.ERROR, None, ("INVALID_VISION_RESULT",))
        reasons = []
        for line, name in ((vision.left, "LEFT"), (vision.right, "RIGHT")):
            if line.observation_status is not ObservationStatus.OBSERVED or not line.verified:
                reasons.append(f"{name}_NOT_OBSERVED_VERIFIED")
        if not vision.time_sync_valid:
            reasons.append("TIME_SYNC_INVALID")
        status = AdmissionStatus.REJECTED if reasons else AdmissionStatus.ACCEPTED
        confidence = min(vision.left.confidence, vision.right.confidence)
        return self._record(run, correlation, "vision", status, vision, tuple(reasons), confidence=confidence,
                            geometry_status="OBSERVED" if not reasons else "UNTRUSTED")

    def _path_record(self, run, correlation, vision, vision_record):
        if vision_record.admission_status is not AdmissionStatus.ACCEPTED:
            return self._record(run, correlation, "path", AdmissionStatus.BLOCKED, vision, ("VISION_NOT_ADMITTED",))
        try:
            result = self.vision_adapter.plan(vision)
        except (TypeError, ValueError, FloatingPointError) as exc:
            return self._record(run, correlation, "path", AdmissionStatus.ERROR, vision,
                                ("VISION_PATH_ADAPTER_ERROR",), (type(exc).__name__,))
        status = AdmissionStatus.ACCEPTED if result.status is VisionPathStatus.VERIFIED else AdmissionStatus.REJECTED
        diagnostics = tuple(result.solver_result.diagnostics.excluded) if result.solver_result else ()
        return self._record(run, correlation, "path", status, vision, result.reason_codes,
                            diagnostics=diagnostics,
                            geometry_status=result.status.value)

    def _blocked_bridge(self, run, correlation, vision):
        return self._record(run, correlation, "vision_to_phase3", AdmissionStatus.BLOCKED, vision,
                            ("VISION_PHASE2_PHASE3_LINK_UNPROVEN",),
                            diagnostics=("No public contract binds VisionPathResult to QualifiedPathResult.",))

    def _motion_record(self, run, correlation, qualified_path, vision):
        if qualified_path is None:
            return self._record(run, correlation, "motion", AdmissionStatus.BLOCKED, vision, ("QUALIFIED_PATH_REQUIRED",))
        try:
            handoff = self.motion_adapter.handoff(qualified_path, mode=HandoffMode.STRICT)
            planned = self.segment_planner.plan(handoff)
        except Exception as exc:
            return self._record(run, correlation, "motion", AdmissionStatus.ERROR, vision,
                                ("MOTION_ADAPTER_ERROR",), (type(exc).__name__,))
        reasons = tuple(code for item in handoff.rejected_segments for code in item.diagnostics) + tuple(
            code for item in planned.rejected_segments for code in item.diagnostics)
        if planned.planned_segments:
            first = planned.planned_segments[0]
            return IntegrationRecord(run, correlation, "motion", AdmissionStatus.ACCEPTED,
                vision.frame_id, first.segment_id, first.timestamp, first.clock_domain, first.coordinate_frame,
                first.units, self.config.config_version, tuple(map(str, first.source_provenance)), None,
                "VALID", first.uncertainty_status.value, first.qualification_status.value,
                (), (), False)
        return self._record(run, correlation, "motion", AdmissionStatus.REJECTED, vision, tuple(dict.fromkeys(reasons or ("MOTION_ADMISSION_REJECTED",))))

    @staticmethod
    def _has_gap(qualified_path):
        if qualified_path is None:
            return True
        return len(getattr(qualified_path, "segments", ())) != 1

    def _record(self, run, correlation, stage, status, vision, reasons=(), diagnostics=(), **extra):
        return IntegrationRecord(run, correlation, stage, status,
            getattr(vision, "frame_id", None), None, self._timestamp(vision),
            getattr(vision, "frame_clock_domain", None), getattr(vision, "coordinate_frame", None),
            getattr(vision, "units", None), self.config.config_version, (),
            extra.get("confidence"), extra.get("geometry_status"), extra.get("uncertainty_status"),
            extra.get("qualification_status"), tuple(dict.fromkeys(reasons)), tuple(diagnostics), False)

    @staticmethod
    def _timestamp(vision):
        value = getattr(vision, "frame_timestamp", None)
        return float(value) if value is not None and np.isfinite(value) else None
