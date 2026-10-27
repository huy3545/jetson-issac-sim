"""Offline Camera Replay Bridge V1.3.1.

Replays recorded camera streams (RGB/Depth) through the 6-module Jetson pipeline
without requiring live Isaac Sim or ROS 2 nodes.

V1.3.1 Hardening:
- Cryptographic tail-truncation protection via final_session_hash in ReplaySessionReport.
- verify_audit_chain with strict length, expected_total_frames, 1-to-1 hash checking, and final session hash verification.
- Guarantees fail-closed security: execution_authorized is NEVER True.
"""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Literal, Sequence
import numpy as np

from data_source import CameraCalibration, RGBDFrame
from integration.camera_frame_contract import CameraFrameContract, validate_frame_contract
from integration.toolhead_closed_loop_harness import (
    ClosedLoopRunResult,
    PipelineStatus,
    ToolheadClosedLoopHarness,
)
from vision.contracts import ObservationStatus
from vision.rgb_perception_adapter import RGBPerceptionAdapter

ReplayMode = Literal["STRICT", "DIAGNOSTIC"]


@dataclass(frozen=True)
class ReplayAuditEvent:
    """Tamper-evident audit event recorded for every single frame in a replay session."""
    session_id: str
    sequence_index: int
    frame_id: str
    rgb_timestamp: float | None
    depth_timestamp: float | None
    clock_domain: str
    source_type: str
    status: str
    reason_codes: tuple[str, ...]
    payload_content_hash: str
    previous_audit_hash: str
    audit_record_hash: str
    diagnostic_only: bool = False


@dataclass(frozen=True)
class ReplayFrameResult:
    """Detailed result of replaying a single camera frame."""
    sequence_index: int
    frame_id: str
    status: str  # "ACCEPTED", "VISION_BLOCKED", "SAFETY_BLOCKED", "CONTRACT_REJECTED", "STREAM_ANOMALY_REJECTED", "ERROR"
    anomalies: tuple[str, ...]
    vision_stage_status: str | None
    safety_stage_status: str | None
    execution_authorized: bool  # Always False
    content_hash: str
    audit_event: ReplayAuditEvent
    diagnostic_only: bool = False
    harness_result: ClosedLoopRunResult | None = None


@dataclass(frozen=True)
class ReplaySessionReport:
    """Comprehensive summary report of an entire replay session with chained audit trail and tail-truncation protection."""
    session_id: str
    stream_id: str
    replay_mode: str
    total_frames: int
    accepted_frames: int
    vision_blocked_frames: int
    safety_blocked_frames: int
    contract_rejected_frames: int
    stream_anomaly_rejected_frames: int
    stream_anomalies: tuple[str, ...] = ()
    frame_anomalies: dict[int, list[str]] = field(default_factory=dict)
    execution_authorized_never_true: bool = True
    audit_trail_hashes: tuple[str, ...] = ()
    audit_events: tuple[ReplayAuditEvent, ...] = ()
    audit_chain_valid: bool = True
    final_session_hash: str = "EMPTY_SESSION"
    summary: str = ""


def compute_payload_content_hash(frame: CameraFrameContract) -> str:
    """Compute deterministic SHA-256 hash of frame input payload."""
    hasher = hashlib.sha256()
    hasher.update(str(frame.sequence_index).encode("utf-8"))
    hasher.update(str(frame.frame_id).encode("utf-8"))
    hasher.update(str(frame.clock_domain).encode("utf-8"))
    hasher.update(str(frame.source_type).encode("utf-8"))
    hasher.update(str(frame.rgb_timestamp).encode("utf-8"))
    hasher.update(str(frame.depth_timestamp).encode("utf-8"))
    hasher.update(str(frame.rgb_depth_registered).encode("utf-8"))

    if frame.rgb_image is not None and isinstance(frame.rgb_image, np.ndarray):
        hasher.update(frame.rgb_image.tobytes())
    else:
        hasher.update(b"NO_RGB_IMAGE")

    if frame.depth_map is not None and isinstance(frame.depth_map, np.ndarray):
        hasher.update(frame.depth_map.tobytes())
    else:
        hasher.update(b"NO_DEPTH_MAP")

    if frame.calibration is not None:
        cal = frame.calibration
        hasher.update(
            f"{cal.fx}_{cal.fy}_{cal.cx}_{cal.cy}_{cal.calibration_status}_{cal.calibration_id}".encode("utf-8")
        )
        hasher.update(np.asarray(cal.camera_to_world, dtype=float).tobytes())
    else:
        hasher.update(b"NO_CALIBRATION")

    if frame.metadata:
        try:
            hasher.update(json.dumps(frame.metadata, sort_keys=True).encode("utf-8"))
        except (TypeError, ValueError):
            hasher.update(str(frame.metadata).encode("utf-8"))

    return hasher.hexdigest()


def compute_audit_record_hash(
    session_id: str,
    sequence_index: int,
    frame_id: str,
    rgb_timestamp: float | None,
    depth_timestamp: float | None,
    clock_domain: str,
    source_type: str,
    status: str,
    reason_codes: Sequence[str],
    payload_content_hash: str,
    previous_audit_hash: str,
    diagnostic_only: bool,
) -> str:
    """Compute chained SHA-256 hash for a ReplayAuditEvent."""
    hasher = hashlib.sha256()
    content = json.dumps(
        {
            "session_id": session_id,
            "sequence_index": sequence_index,
            "frame_id": frame_id,
            "rgb_timestamp": rgb_timestamp,
            "depth_timestamp": depth_timestamp,
            "clock_domain": clock_domain,
            "source_type": source_type,
            "status": status,
            "reason_codes": sorted(list(reason_codes)),
            "payload_content_hash": payload_content_hash,
            "previous_audit_hash": previous_audit_hash,
            "diagnostic_only": diagnostic_only,
        },
        sort_keys=True,
    )
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def compute_final_session_hash(session_id: str, total_frames: int, last_audit_record_hash: str | None) -> str:
    """Compute final session hash locking the chain against tail-truncation."""
    if total_frames <= 0 or not last_audit_record_hash:
        return "EMPTY_SESSION"
    hasher = hashlib.sha256()
    hasher.update(f"{session_id}{total_frames}{last_audit_record_hash}".encode("utf-8"))
    return hasher.hexdigest()


def verify_audit_chain(report: ReplaySessionReport, expected_total_frames: int | None = None) -> bool:
    """Verify cryptographic chaining integrity of all audit events in a report with tail-truncation protection."""
    # 1. Total frames boundary checks
    if report.total_frames < 0:
        return False

    if len(report.audit_events) != report.total_frames or len(report.audit_trail_hashes) != report.total_frames:
        return False

    if expected_total_frames is not None and len(report.audit_events) != expected_total_frames:
        return False

    # 2. Empty session handling
    if report.total_frames == 0:
        return report.final_session_hash == "EMPTY_SESSION"

    # If total_frames > 0, audit_events MUST NOT be empty
    if not report.audit_events:
        return False

    # 3. Step-by-step cryptographic chain verification
    expected_prev = "GENESIS"
    for i, event in enumerate(report.audit_events):
        # 1-to-1 match with audit_trail_hashes
        if event.audit_record_hash != report.audit_trail_hashes[i]:
            return False

        if event.previous_audit_hash != expected_prev:
            return False

        recalculated = compute_audit_record_hash(
            session_id=event.session_id,
            sequence_index=event.sequence_index,
            frame_id=event.frame_id,
            rgb_timestamp=event.rgb_timestamp,
            depth_timestamp=event.depth_timestamp,
            clock_domain=event.clock_domain,
            source_type=event.source_type,
            status=event.status,
            reason_codes=event.reason_codes,
            payload_content_hash=event.payload_content_hash,
            previous_audit_hash=event.previous_audit_hash,
            diagnostic_only=event.diagnostic_only,
        )
        if event.audit_record_hash != recalculated:
            return False

        expected_prev = event.audit_record_hash

    # 4. Final session hash verification (protects against removing events from the tail)
    expected_final_hash = compute_final_session_hash(
        report.session_id, report.total_frames, report.audit_events[-1].audit_record_hash
    )
    if report.final_session_hash != expected_final_hash:
        return False

    return True


class OfflineCameraReplay:
    """Executes offline frame sequence replay through perception and closed loop harness."""

    def __init__(
        self,
        adapter: RGBPerceptionAdapter | None = None,
        harness: ToolheadClosedLoopHarness | None = None,
        allow_synthetic_dry_run: bool = True,
        replay_mode: ReplayMode = "STRICT",
    ):
        self.adapter = adapter or RGBPerceptionAdapter()
        self.harness = harness or ToolheadClosedLoopHarness()
        self.allow_synthetic_dry_run = allow_synthetic_dry_run
        self.replay_mode: ReplayMode = replay_mode
        self.last_sequence_index: int | None = None
        self.last_timestamp: float | None = None
        self.last_audit_hash: str = "GENESIS"

    def reset_stream_state(self) -> None:
        """Reset sequence tracking between stream runs."""
        self.last_sequence_index = None
        self.last_timestamp = None
        self.last_audit_hash = "GENESIS"

    def replay_frame(
        self,
        frame: CameraFrameContract,
        session_id: str = "default-session",
    ) -> ReplayFrameResult:
        """Process a single frame contract through the 6-module pipeline with chained audit."""
        anomalies: list[str] = []
        stream_anomalies: list[str] = []

        # 1. Stream-level anomaly checks
        if self.last_sequence_index is not None:
            expected_seq = self.last_sequence_index + 1
            if frame.sequence_index != expected_seq:
                stream_anomalies.append(
                    f"FRAME_GAP_DETECTED: expected {expected_seq}, got {frame.sequence_index}"
                )

        if self.last_timestamp is not None and frame.rgb_timestamp is not None:
            if frame.rgb_timestamp <= self.last_timestamp:
                stream_anomalies.append(
                    f"NON_MONOTONIC_TIMESTAMP: current {frame.rgb_timestamp:.6f} <= previous {self.last_timestamp:.6f}"
                )

        # Update stream trackers
        self.last_sequence_index = frame.sequence_index
        if frame.rgb_timestamp is not None:
            self.last_timestamp = frame.rgb_timestamp

        anomalies.extend(stream_anomalies)
        has_stream_anomaly = len(stream_anomalies) > 0

        # Strict vs Diagnostic Mode Branching
        diagnostic_only = False
        if has_stream_anomaly:
            if self.replay_mode == "STRICT":
                # Strict mode rejects processing immediately
                status = "STREAM_ANOMALY_REJECTED"
                payload_hash = compute_payload_content_hash(frame)
                audit_hash = compute_audit_record_hash(
                    session_id=session_id,
                    sequence_index=frame.sequence_index,
                    frame_id=frame.frame_id,
                    rgb_timestamp=frame.rgb_timestamp,
                    depth_timestamp=frame.depth_timestamp,
                    clock_domain=frame.clock_domain,
                    source_type=frame.source_type,
                    status=status,
                    reason_codes=anomalies,
                    payload_content_hash=payload_hash,
                    previous_audit_hash=self.last_audit_hash,
                    diagnostic_only=False,
                )
                audit_event = ReplayAuditEvent(
                    session_id=session_id,
                    sequence_index=frame.sequence_index,
                    frame_id=frame.frame_id,
                    rgb_timestamp=frame.rgb_timestamp,
                    depth_timestamp=frame.depth_timestamp,
                    clock_domain=frame.clock_domain,
                    source_type=frame.source_type,
                    status=status,
                    reason_codes=tuple(anomalies),
                    payload_content_hash=payload_hash,
                    previous_audit_hash=self.last_audit_hash,
                    audit_record_hash=audit_hash,
                    diagnostic_only=False,
                )
                self.last_audit_hash = audit_hash

                return ReplayFrameResult(
                    sequence_index=frame.sequence_index,
                    frame_id=frame.frame_id,
                    status=status,
                    anomalies=tuple(anomalies),
                    vision_stage_status=None,
                    safety_stage_status=None,
                    execution_authorized=False,
                    content_hash=audit_hash,
                    audit_event=audit_event,
                    diagnostic_only=False,
                    harness_result=None,
                )
            else:
                # Diagnostic mode continues processing for telemetry
                diagnostic_only = True

        # 2. Contract validation
        valid, contract_errors = validate_frame_contract(frame)
        if not valid:
            anomalies.extend(contract_errors)
            status = "CONTRACT_REJECTED"
            payload_hash = compute_payload_content_hash(frame)
            audit_hash = compute_audit_record_hash(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=anomalies,
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                diagnostic_only=diagnostic_only,
            )
            audit_event = ReplayAuditEvent(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=tuple(anomalies),
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                audit_record_hash=audit_hash,
                diagnostic_only=diagnostic_only,
            )
            self.last_audit_hash = audit_hash

            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status=status,
                anomalies=tuple(anomalies),
                vision_stage_status=None,
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=audit_hash,
                audit_event=audit_event,
                diagnostic_only=diagnostic_only,
                harness_result=None,
            )

        # Check calibration status anomaly
        if frame.calibration is not None:
            if frame.calibration.calibration_status in ("SYNTHETIC_TEST_ONLY", "REGISTERED"):
                anomalies.append(f"CALIBRATION_{frame.calibration.calibration_status}")
        else:
            anomalies.append("MISSING_CALIBRATION")
            status = "CONTRACT_REJECTED"
            payload_hash = compute_payload_content_hash(frame)
            audit_hash = compute_audit_record_hash(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=anomalies,
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                diagnostic_only=diagnostic_only,
            )
            audit_event = ReplayAuditEvent(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=tuple(anomalies),
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                audit_record_hash=audit_hash,
                diagnostic_only=diagnostic_only,
            )
            self.last_audit_hash = audit_hash

            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status=status,
                anomalies=tuple(anomalies),
                vision_stage_status=None,
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=audit_hash,
                audit_event=audit_event,
                diagnostic_only=diagnostic_only,
                harness_result=None,
            )

        # 3. Process via Perception Adapter
        try:
            metric_result, quality = self.adapter.process_rgbd(
                rgb=frame.rgb_image,
                depth=frame.depth_map,
                calibration=frame.calibration,
                timestamp_s=frame.rgb_timestamp or 0.0,
                frame_id=frame.frame_id,
                clock_domain=frame.clock_domain,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                rgb_depth_registered=frame.rgb_depth_registered,
                allow_synthetic_dry_run=self.allow_synthetic_dry_run,
            )
        except ValueError as exc:
            anomalies.append(str(exc))
            status = "VISION_BLOCKED"
            payload_hash = compute_payload_content_hash(frame)
            audit_hash = compute_audit_record_hash(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=anomalies,
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                diagnostic_only=diagnostic_only,
            )
            audit_event = ReplayAuditEvent(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=tuple(anomalies),
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                audit_record_hash=audit_hash,
                diagnostic_only=diagnostic_only,
            )
            self.last_audit_hash = audit_hash

            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status=status,
                anomalies=tuple(anomalies),
                vision_stage_status="ERROR",
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=audit_hash,
                audit_event=audit_event,
                diagnostic_only=diagnostic_only,
                harness_result=None,
            )

        # If Vision observed status is not valid
        if metric_result.left.observation_status is not ObservationStatus.OBSERVED:
            anomalies.extend(quality.diagnostics)
            status = "VISION_BLOCKED"
            payload_hash = compute_payload_content_hash(frame)
            audit_hash = compute_audit_record_hash(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=anomalies,
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                diagnostic_only=diagnostic_only,
            )
            audit_event = ReplayAuditEvent(
                session_id=session_id,
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                clock_domain=frame.clock_domain,
                source_type=frame.source_type,
                status=status,
                reason_codes=tuple(anomalies),
                payload_content_hash=payload_hash,
                previous_audit_hash=self.last_audit_hash,
                audit_record_hash=audit_hash,
                diagnostic_only=diagnostic_only,
            )
            self.last_audit_hash = audit_hash

            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status=status,
                anomalies=tuple(anomalies),
                vision_stage_status="INVALID",
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=audit_hash,
                audit_event=audit_event,
                diagnostic_only=diagnostic_only,
                harness_result=None,
            )

        # 4. Feed into 6-Module Closed-Loop Harness
        run_res = self.harness.run(
            metric_result,
            run_id=f"replay-f{frame.sequence_index}-{frame.frame_id}",
        )

        # Must strictly enforce fail-closed
        if run_res.execution_authorized:
            raise AssertionError("SECURITY VIOLATION: execution_authorized MUST NEVER be True!")

        vision_status = run_res.stage_records[0].status.value
        safety_status = run_res.stage_records[3].status.value

        is_workspace_violation = any(
            "WORKSPACE_VIOLATION" in r for r in run_res.stage_records[3].reason_codes
        )
        if is_workspace_violation:
            anomalies.append("WORKSPACE_VIOLATION")
            final_status = "SAFETY_BLOCKED"
        else:
            final_status = "ACCEPTED"

        payload_hash = compute_payload_content_hash(frame)
        audit_hash = compute_audit_record_hash(
            session_id=session_id,
            sequence_index=frame.sequence_index,
            frame_id=frame.frame_id,
            rgb_timestamp=frame.rgb_timestamp,
            depth_timestamp=frame.depth_timestamp,
            clock_domain=frame.clock_domain,
            source_type=frame.source_type,
            status=final_status,
            reason_codes=anomalies,
            payload_content_hash=payload_hash,
            previous_audit_hash=self.last_audit_hash,
            diagnostic_only=diagnostic_only,
        )
        audit_event = ReplayAuditEvent(
            session_id=session_id,
            sequence_index=frame.sequence_index,
            frame_id=frame.frame_id,
            rgb_timestamp=frame.rgb_timestamp,
            depth_timestamp=frame.depth_timestamp,
            clock_domain=frame.clock_domain,
            source_type=frame.source_type,
            status=final_status,
            reason_codes=tuple(anomalies),
            payload_content_hash=payload_hash,
            previous_audit_hash=self.last_audit_hash,
            audit_record_hash=audit_hash,
            diagnostic_only=diagnostic_only,
        )
        self.last_audit_hash = audit_hash

        return ReplayFrameResult(
            sequence_index=frame.sequence_index,
            frame_id=frame.frame_id,
            status=final_status,
            anomalies=tuple(anomalies),
            vision_stage_status=vision_status,
            safety_stage_status=safety_status,
            execution_authorized=False,
            content_hash=audit_hash,
            audit_event=audit_event,
            diagnostic_only=diagnostic_only,
            harness_result=run_res,
        )

    def replay_stream(
        self,
        frames: Sequence[CameraFrameContract],
        session_id: str = "replay-session-001",
        stream_id: str = "camera-stream-main",
    ) -> ReplaySessionReport:
        """Replay an entire sequence of frames and produce a session report with chained audit."""
        self.reset_stream_state()

        accepted = 0
        vision_blocked = 0
        safety_blocked = 0
        contract_rejected = 0
        stream_anomaly_rejected = 0
        frame_anomalies: dict[int, list[str]] = {}
        all_stream_anomalies: list[str] = []
        audit_hashes: list[str] = []
        audit_events: list[ReplayAuditEvent] = []
        all_unauthorized = True

        for frame in frames:
            res = self.replay_frame(frame, session_id=session_id)
            if res.execution_authorized:
                all_unauthorized = False

            audit_hashes.append(res.content_hash)
            audit_events.append(res.audit_event)

            if res.anomalies:
                frame_anomalies[frame.sequence_index] = list(res.anomalies)
                for a in res.anomalies:
                    if a not in all_stream_anomalies:
                        all_stream_anomalies.append(a)

            if res.status == "ACCEPTED":
                accepted += 1
            elif res.status == "VISION_BLOCKED":
                vision_blocked += 1
            elif res.status == "SAFETY_BLOCKED":
                safety_blocked += 1
            elif res.status == "CONTRACT_REJECTED":
                contract_rejected += 1
            elif res.status == "STREAM_ANOMALY_REJECTED":
                stream_anomaly_rejected += 1

        summary = (
            f"Session {session_id} on stream {stream_id} (Mode: {self.replay_mode}): "
            f"{len(frames)} total frames, {accepted} accepted, "
            f"{vision_blocked} vision blocked, {safety_blocked} safety blocked, "
            f"{contract_rejected} contract rejected, {stream_anomaly_rejected} stream anomaly rejected. "
            f"Execution authorized: NEVER (Gate B Fail-Closed)."
        )

        last_hash = audit_events[-1].audit_record_hash if audit_events else None
        final_hash = compute_final_session_hash(session_id, len(frames), last_hash)

        report = ReplaySessionReport(
            session_id=session_id,
            stream_id=stream_id,
            replay_mode=self.replay_mode,
            total_frames=len(frames),
            accepted_frames=accepted,
            vision_blocked_frames=vision_blocked,
            safety_blocked_frames=safety_blocked,
            contract_rejected_frames=contract_rejected,
            stream_anomaly_rejected_frames=stream_anomaly_rejected,
            stream_anomalies=tuple(all_stream_anomalies),
            frame_anomalies=frame_anomalies,
            execution_authorized_never_true=all_unauthorized,
            audit_trail_hashes=tuple(audit_hashes),
            audit_events=tuple(audit_events),
            audit_chain_valid=True,
            final_session_hash=final_hash,
            summary=summary,
        )

        chain_ok = verify_audit_chain(report)
        if not chain_ok:
            object.__setattr__(report, "audit_chain_valid", False)

        return report
