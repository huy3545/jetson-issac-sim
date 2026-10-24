"""Offline Camera Replay Bridge (Option C).

Replays recorded camera streams (RGB/Depth) through the 6-module Jetson pipeline
without requiring live Isaac Sim or ROS 2 nodes.
Enforces strict fail-closed security: execution_authorized is NEVER True.
"""
from dataclasses import dataclass, field
from typing import Sequence
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


@dataclass(frozen=True)
class ReplayFrameResult:
    """Detailed result of replaying a single camera frame."""
    sequence_index: int
    frame_id: str
    status: str  # "ACCEPTED", "VISION_BLOCKED", "SAFETY_BLOCKED", "CONTRACT_REJECTED", "ERROR"
    anomalies: tuple[str, ...]
    vision_stage_status: str | None
    safety_stage_status: str | None
    execution_authorized: bool  # Always False
    content_hash: str | None
    harness_result: ClosedLoopRunResult | None = None


@dataclass(frozen=True)
class ReplaySessionReport:
    """Comprehensive summary report of an entire replay session."""
    session_id: str
    stream_id: str
    total_frames: int
    accepted_frames: int
    vision_blocked_frames: int
    safety_blocked_frames: int
    contract_rejected_frames: int
    stream_anomalies: tuple[str, ...] = ()
    frame_anomalies: dict[int, list[str]] = field(default_factory=dict)
    execution_authorized_never_true: bool = True
    audit_trail_hashes: tuple[str, ...] = ()
    summary: str = ""


class OfflineCameraReplay:
    """Executes offline frame sequence replay through perception and closed loop harness."""

    def __init__(
        self,
        adapter: RGBPerceptionAdapter | None = None,
        harness: ToolheadClosedLoopHarness | None = None,
        allow_synthetic_dry_run: bool = True,
    ):
        self.adapter = adapter or RGBPerceptionAdapter()
        self.harness = harness or ToolheadClosedLoopHarness()
        self.allow_synthetic_dry_run = allow_synthetic_dry_run
        self.last_sequence_index: int | None = None
        self.last_timestamp: float | None = None

    def reset_stream_state(self) -> None:
        """Reset sequence tracking between stream runs."""
        self.last_sequence_index = None
        self.last_timestamp = None

    def replay_frame(self, frame: CameraFrameContract) -> ReplayFrameResult:
        """Process a single frame contract through the 6-module pipeline."""
        anomalies: list[str] = []

        # 1. Stream-level anomaly checks
        if self.last_sequence_index is not None:
            expected_seq = self.last_sequence_index + 1
            if frame.sequence_index != expected_seq:
                anomalies.append(f"FRAME_GAP_DETECTED: expected {expected_seq}, got {frame.sequence_index}")

        if self.last_timestamp is not None and frame.rgb_timestamp is not None:
            if frame.rgb_timestamp <= self.last_timestamp:
                anomalies.append(
                    f"NON_MONOTONIC_TIMESTAMP: current {frame.rgb_timestamp:.6f} <= previous {self.last_timestamp:.6f}"
                )

        # Update stream trackers
        self.last_sequence_index = frame.sequence_index
        if frame.rgb_timestamp is not None:
            self.last_timestamp = frame.rgb_timestamp

        # 2. Contract validation
        valid, contract_errors = validate_frame_contract(frame)
        if not valid:
            anomalies.extend(contract_errors)
            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status="CONTRACT_REJECTED",
                anomalies=tuple(anomalies),
                vision_stage_status=None,
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=None,
                harness_result=None,
            )

        # Check calibration status anomaly
        if frame.calibration is not None:
            if frame.calibration.calibration_status == "SYNTHETIC_TEST_ONLY":
                anomalies.append("CALIBRATION_SYNTHETIC_TEST_ONLY")
        else:
            anomalies.append("MISSING_CALIBRATION")
            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status="CONTRACT_REJECTED",
                anomalies=tuple(anomalies),
                vision_stage_status=None,
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=None,
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
            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status="VISION_BLOCKED",
                anomalies=tuple(anomalies),
                vision_stage_status="ERROR",
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=None,
                harness_result=None,
            )

        # If Vision observed status is not valid
        if metric_result.left.observation_status is not ObservationStatus.OBSERVED:
            anomalies.extend(quality.diagnostics)
            return ReplayFrameResult(
                sequence_index=frame.sequence_index,
                frame_id=frame.frame_id,
                status="VISION_BLOCKED",
                anomalies=tuple(anomalies),
                vision_stage_status="INVALID",
                safety_stage_status=None,
                execution_authorized=False,
                content_hash=None,
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

        # Determine overall status:
        # In dry-run simulation, harness accepts Vision, Path, Motion, Logs,
        # but Safety is always BLOCKED by STRICT_GATE_B_BLOCKED policy.
        # If no workspace violation, it is considered ACCEPTED dry-run frame.
        is_workspace_violation = any(
            "WORKSPACE_VIOLATION" in r for r in run_res.stage_records[3].reason_codes
        )
        if is_workspace_violation:
            anomalies.append("WORKSPACE_VIOLATION")
            final_status = "SAFETY_BLOCKED"
        else:
            final_status = "ACCEPTED"

        last_hash = run_res.stage_records[-1].content_hash

        return ReplayFrameResult(
            sequence_index=frame.sequence_index,
            frame_id=frame.frame_id,
            status=final_status,
            anomalies=tuple(anomalies),
            vision_stage_status=vision_status,
            safety_stage_status=safety_status,
            execution_authorized=False,
            content_hash=last_hash,
            harness_result=run_res,
        )

    def replay_stream(
        self,
        frames: Sequence[CameraFrameContract],
        session_id: str = "replay-session-001",
        stream_id: str = "camera-stream-main",
    ) -> ReplaySessionReport:
        """Replay an entire sequence of frames and produce a session report."""
        self.reset_stream_state()

        accepted = 0
        vision_blocked = 0
        safety_blocked = 0
        contract_rejected = 0
        frame_anomalies: dict[int, list[str]] = {}
        all_stream_anomalies: list[str] = []
        audit_hashes: list[str] = []
        all_unauthorized = True

        for frame in frames:
            res = self.replay_frame(frame)
            if res.execution_authorized:
                all_unauthorized = False

            if res.content_hash:
                audit_hashes.append(res.content_hash)

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

        summary = (
            f"Session {session_id} on stream {stream_id}: "
            f"{len(frames)} total frames, {accepted} accepted, "
            f"{vision_blocked} vision blocked, {safety_blocked} safety blocked, "
            f"{contract_rejected} contract rejected. "
            f"Execution authorized: NEVER (Gate B Fail-Closed)."
        )

        return ReplaySessionReport(
            session_id=session_id,
            stream_id=stream_id,
            total_frames=len(frames),
            accepted_frames=accepted,
            vision_blocked_frames=vision_blocked,
            safety_blocked_frames=safety_blocked,
            contract_rejected_frames=contract_rejected,
            stream_anomalies=tuple(all_stream_anomalies),
            frame_anomalies=frame_anomalies,
            execution_authorized_never_true=all_unauthorized,
            audit_trail_hashes=tuple(audit_hashes),
            summary=summary,
        )
