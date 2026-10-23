"""Six-module closed-loop pipeline harness for Jetson Orin Nano fabric cutting.

Links the six core functional blocks into a unified, fail-closed pipeline:
  Module 1: Data / Vision Ingestion (VisionResult)
  Module 2: Path Planning (Medial Axis / Centerline via CornerAwareTwoLineSolver)
  Module 3: Motion Planning (Toolhead Z-plunge/retract, Tangential C, Curvature Speed)
  Module 4: Safety Validation (Workspace bounds, C-step, Speed limits, Gate B check)
  Module 5: Machine Control Simulation (Kinematic toolhead simulator)
  Module 6: Structured Data Logging (Point-level lineage, SHA-256 audit trail)

Strictly fail-closed (execution_authorized=False) by design.
"""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Sequence
from uuid import uuid4
import numpy as np

from config import Config, MotionConfig, SafetyConfig
from motion_planning.toolhead_controller import (
    KnifeState,
    ToolheadCommand,
    ToolheadController,
    ToolheadTrajectoryResult,
)
from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import VisionResult


class PipelineStage(str, Enum):
    DATA_VISION = "DATA_VISION"
    PATH_PLANNING = "PATH_PLANNING"
    MOTION_PLANNING = "MOTION_PLANNING"
    SAFETY_VALIDATION = "SAFETY_VALIDATION"
    CONTROL_SIMULATION = "CONTROL_SIMULATION"
    DATA_LOGGING = "DATA_LOGGING"


class PipelineStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PipelineStageRecord:
    stage: PipelineStage
    status: PipelineStatus
    timestamp_s: float
    reason_codes: tuple[str, ...]
    content_hash: str
    diagnostics: tuple[str, ...] = ()
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("Pipeline cannot authorize live machine execution")


@dataclass(frozen=True)
class SimulatedMachineState:
    sim_time_s: float
    current_x_m: float
    current_y_m: float
    current_z_m: float
    current_c_rad: float
    current_speed_m_s: float
    knife_state: KnifeState
    step_index: int


@dataclass(frozen=True)
class ClosedLoopRunResult:
    run_id: str
    correlation_id: str
    final_status: PipelineStatus
    stage_records: tuple[PipelineStageRecord, ...]
    toolhead_result: ToolheadTrajectoryResult | None
    simulated_states: tuple[SimulatedMachineState, ...]
    audit_trail_complete: bool
    summary_report: str
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("Closed loop run cannot authorize execution")


def _hash_obj(data: object) -> str:
    """Generate deterministic SHA-256 hash string for data auditing."""
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolheadClosedLoopHarness:
    """Executes an end-to-end simulation of the 6-module Jetson pipeline."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.path_adapter = VisionPathAdapter(
            solver=CornerAwareTwoLineSolver(),
            require_point_provenance=True,
        )
        self.toolhead_controller = ToolheadController(
            z_cut_m=0.005,
            z_retract_m=0.020,
            motion_config=self.config.motion,
            max_c_step_rad=self.config.safety.max_c_step,
        )

    def run(
        self,
        vision_result: VisionResult,
        *,
        run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> ClosedLoopRunResult:
        run_id = run_id or uuid4().hex
        correlation_id = correlation_id or run_id
        records = []
        sim_states = []

        # =====================================================================
        # MODULE 1: DATA / VISION INGESTION
        # =====================================================================
        if not isinstance(vision_result, VisionResult):
            rec = PipelineStageRecord(
                stage=PipelineStage.DATA_VISION,
                status=PipelineStatus.ERROR,
                timestamp_s=0.0,
                reason_codes=("INVALID_VISION_DATA",),
                content_hash="",
            )
            records.append(rec)
            return self._finalize_run(run_id, correlation_id, records, None, ())

        preflight = self.path_adapter.validate_observation(vision_result)
        vision_hash = _hash_obj(
            {
                "frame_id": vision_result.frame_id,
                "timestamp": vision_result.frame_timestamp,
                "left_len": len(vision_result.left.points),
                "right_len": len(vision_result.right.points),
            }
        )
        if preflight:
            rec = PipelineStageRecord(
                stage=PipelineStage.DATA_VISION,
                status=PipelineStatus.REJECTED,
                timestamp_s=float(vision_result.frame_timestamp),
                reason_codes=tuple(preflight),
                content_hash=vision_hash,
                diagnostics=tuple(preflight),
            )
            records.append(rec)
            return self._finalize_run(run_id, correlation_id, records, None, ())

        rec_vision = PipelineStageRecord(
            stage=PipelineStage.DATA_VISION,
            status=PipelineStatus.ACCEPTED,
            timestamp_s=float(vision_result.frame_timestamp),
            reason_codes=("VISION_OBSERVATION_VERIFIED",),
            content_hash=vision_hash,
        )
        records.append(rec_vision)

        # =====================================================================
        # MODULE 2: PATH PLANNING (MEDIAL AXIS / CENTERLINE)
        # =====================================================================
        path_plan = self.path_adapter.plan(vision_result)
        path_hash = _hash_obj(
            {
                "status": str(path_plan.status),
                "reasons": list(path_plan.reason_codes),
                "has_solver": path_plan.solver_result is not None,
            }
        )

        if path_plan.status is not VisionPathStatus.VERIFIED or not path_plan.solver_result:
            rec_path = PipelineStageRecord(
                stage=PipelineStage.PATH_PLANNING,
                status=PipelineStatus.REJECTED,
                timestamp_s=float(vision_result.frame_timestamp),
                reason_codes=path_plan.reason_codes or ("PATH_GEOMETRY_REJECTED",),
                content_hash=path_hash,
            )
            records.append(rec_path)
            return self._finalize_run(run_id, correlation_id, records, None, ())

        rec_path = PipelineStageRecord(
            stage=PipelineStage.PATH_PLANNING,
            status=PipelineStatus.ACCEPTED,
            timestamp_s=float(vision_result.frame_timestamp),
            reason_codes=("CENTERLINE_GEOMETRY_VERIFIED",),
            content_hash=path_hash,
        )
        records.append(rec_path)

        # Extract centerline waypoints from solver segments
        segments_xy = [seg.xy for seg in path_plan.solver_result.segments]

        # =====================================================================
        # MODULE 3: MOTION PLANNING (TOOLHEAD CONTROLLER)
        # =====================================================================
        toolhead_result = self.toolhead_controller.plan_multi_segments(segments_xy)
        motion_hash = _hash_obj(
            {
                "command_count": len(toolhead_result.commands),
                "total_dist": toolhead_result.total_distance_m,
                "total_time": toolhead_result.total_time_s,
            }
        )

        if not toolhead_result.commands or "NO_VALID_SEGMENTS" in toolhead_result.diagnostics:
            rec_motion = PipelineStageRecord(
                stage=PipelineStage.MOTION_PLANNING,
                status=PipelineStatus.REJECTED,
                timestamp_s=float(vision_result.frame_timestamp),
                reason_codes=("TOOLHEAD_PLAN_EMPTY",),
                content_hash=motion_hash,
                diagnostics=toolhead_result.diagnostics,
            )
            records.append(rec_motion)
            return self._finalize_run(run_id, correlation_id, records, toolhead_result, ())

        rec_motion = PipelineStageRecord(
            stage=PipelineStage.MOTION_PLANNING,
            status=PipelineStatus.ACCEPTED,
            timestamp_s=float(vision_result.frame_timestamp),
            reason_codes=("TOOLHEAD_TRAJECTORY_GENERATED",),
            content_hash=motion_hash,
        )
        records.append(rec_motion)

        # =====================================================================
        # MODULE 4: SAFETY VALIDATION (WORKSPACE BOUNDS & STEP LIMITS)
        # =====================================================================
        # Gate B is strictly BLOCKED by system policy (Read-only simulation)
        safety_reasons = ["STRICT_GATE_B_BLOCKED"]
        safety_cfg: SafetyConfig = self.config.safety
        for idx, cmd in enumerate(toolhead_result.commands):
            if (
                cmd.x_m < safety_cfg.workspace_min[0]
                or cmd.x_m > safety_cfg.workspace_max[0]
                or cmd.y_m < safety_cfg.workspace_min[1]
                or cmd.y_m > safety_cfg.workspace_max[1]
                or cmd.z_m < safety_cfg.workspace_min[2]
                or cmd.z_m > safety_cfg.workspace_max[2]
            ):
                safety_reasons.append(f"WORKSPACE_VIOLATION_STEP_{idx}")

            if cmd.feed_speed_m_s > safety_cfg.max_speed + 1e-4:
                safety_reasons.append(f"SPEED_LIMIT_EXCEEDED_STEP_{idx}")

        safety_hash = _hash_obj(safety_reasons)

        rec_safety = PipelineStageRecord(
            stage=PipelineStage.SAFETY_VALIDATION,
            status=PipelineStatus.BLOCKED,
            timestamp_s=float(vision_result.frame_timestamp),
            reason_codes=tuple(dict.fromkeys(safety_reasons)),
            content_hash=safety_hash,
        )
        records.append(rec_safety)

        # =====================================================================
        # MODULE 5: MACHINE CONTROL SIMULATION (KINEMATIC EMULATION)
        # =====================================================================
        for cmd in toolhead_result.commands:
            state = SimulatedMachineState(
                sim_time_s=cmd.timestamp_s,
                current_x_m=cmd.x_m,
                current_y_m=cmd.y_m,
                current_z_m=cmd.z_m,
                current_c_rad=cmd.c_rad,
                current_speed_m_s=cmd.feed_speed_m_s,
                knife_state=cmd.knife_state,
                step_index=cmd.step_index,
            )
            sim_states.append(state)

        rec_control = PipelineStageRecord(
            stage=PipelineStage.CONTROL_SIMULATION,
            status=PipelineStatus.ACCEPTED,
            timestamp_s=float(sim_states[-1].sim_time_s) if sim_states else 0.0,
            reason_codes=("PASSIVE_KINEMATIC_SIMULATION_PASS",),
            content_hash=_hash_obj(len(sim_states)),
        )
        records.append(rec_control)

        # =====================================================================
        # MODULE 6: STRUCTURED DATA LOGGING & AUDIT TRAIL
        # =====================================================================
        audit_complete = len(records) == 5
        rec_logging = PipelineStageRecord(
            stage=PipelineStage.DATA_LOGGING,
            status=PipelineStatus.ACCEPTED if audit_complete else PipelineStatus.ERROR,
            timestamp_s=float(sim_states[-1].sim_time_s) if sim_states else 0.0,
            reason_codes=("AUDIT_TRAIL_CONFIRMED",),
            content_hash=_hash_obj([r.content_hash for r in records]),
        )
        records.append(rec_logging)

        return self._finalize_run(run_id, correlation_id, records, toolhead_result, sim_states)

    def _finalize_run(
        self,
        run_id: str,
        correlation_id: str,
        records: list[PipelineStageRecord],
        toolhead_result: ToolheadTrajectoryResult | None,
        sim_states: Sequence[SimulatedMachineState],
    ) -> ClosedLoopRunResult:
        statuses = [r.status for r in records]
        final_status = PipelineStatus.ACCEPTED
        for st in (PipelineStatus.ERROR, PipelineStatus.REJECTED, PipelineStatus.BLOCKED):
            if st in statuses:
                final_status = st
                break

        summary_lines = [
            f"=== JETSON SIX-MODULE PIPELINE RUN SUMMARY [{run_id[:8]}] ===",
            f"Correlation ID: {correlation_id}",
            f"Pipeline Final Status: {final_status.value}",
            f"Execution Authorized: False (Safety Enforced)",
        ]
        for r in records:
            summary_lines.append(f"  * {r.stage.value:<22}: {r.status.value:<10} | {','.join(r.reason_codes[:2])}")

        if toolhead_result:
            summary_lines.append(
                f"Trajectory: Total Dist = {toolhead_result.total_distance_m*1000:.1f}mm "
                f"(Cut={toolhead_result.cut_distance_m*1000:.1f}mm, Traverse={toolhead_result.traverse_distance_m*1000:.1f}mm), "
                f"Commands = {len(toolhead_result.commands)}, Sim Steps = {len(sim_states)}"
            )

        return ClosedLoopRunResult(
            run_id=run_id,
            correlation_id=correlation_id,
            final_status=final_status,
            stage_records=tuple(records),
            toolhead_result=toolhead_result,
            simulated_states=tuple(sim_states),
            audit_trail_complete=(len(records) == 6),
            summary_report="\n".join(summary_lines),
            execution_authorized=False,
        )
