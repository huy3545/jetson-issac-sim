"""Toolhead motion controller for knife Z-axis and tangential C-axis.

Responsible for:
1. Z-axis stroke management: Plunging to Z_cut (cutting depth) and retracting to
   Z_retract (clearance height) across discontinuities to avoid tearing fabric.
2. Tangential C-axis orientation: Continuous tangent tracking C = arctan2(dy, dx)
   with unwrapping and pre-orientation in the air before plunging.
3. Curvature-adaptive speed profiling: Automatically decelerating in sharp turns
   to preserve cut accuracy and blade mechanical integrity.

Strictly non-executable (execution_authorized=False) by design.
"""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence
import numpy as np

from config import MotionConfig


class KnifeState(str, Enum):
    RETRACTED = "RETRACTED"
    PLUNGING = "PLUNGING"
    CUTTING = "CUTTING"
    RETRACTING = "RETRACTING"
    TRAVERSING = "TRAVERSING"


@dataclass(frozen=True)
class ToolheadCommand:
    """Individual waypoint command for knife toolhead."""
    step_index: int
    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float
    c_rad: float
    feed_speed_m_s: float
    knife_state: KnifeState
    curvature_m_inv: float
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("Toolhead controller cannot authorize live execution")


@dataclass(frozen=True)
class ToolheadTrajectoryResult:
    commands: tuple[ToolheadCommand, ...]
    total_distance_m: float
    cut_distance_m: float
    traverse_distance_m: float
    total_time_s: float
    cut_count: int
    max_speed_m_s: float
    z_cut_m: float
    z_retract_m: float
    diagnostics: tuple[str, ...]
    execution_authorized: bool = False


class ToolheadController:
    """Calculates toolhead kinematic profiles (X, Y, Z, C, speed)."""

    def __init__(
        self,
        z_cut_m: float = 0.005,
        z_retract_m: float = 0.020,
        motion_config: MotionConfig | None = None,
        max_c_step_rad: float = 0.5,
    ):
        self.z_cut_m = float(z_cut_m)
        self.z_retract_m = float(z_retract_m)
        cfg = motion_config or MotionConfig()
        self.max_speed = float(cfg.max_speed)
        self.acceleration = float(cfg.acceleration)
        self.lateral_acceleration = float(cfg.lateral_acceleration)
        self.max_c_step_rad = float(max_c_step_rad)
        self.z_plunge_speed = 0.02  # m/s
        self.z_retract_speed = 0.04  # m/s

    def compute_tangent_and_curvature(
        self, points_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute cumulative distance s, tangent angle c_rad (unwrapped), and curvature kappa."""
        pts = np.asarray(points_xy, dtype=float)
        n = len(pts)
        if n < 2:
            raise ValueError("At least two points required for tangent calculation")

        diffs = np.diff(pts, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        if (ds <= 1e-9).any():
            # If duplicated adjacent points exist, add tiny epsilon
            ds = np.maximum(ds, 1e-6)

        s = np.r_[0.0, np.cumsum(ds)]
        tangents = np.gradient(pts, s, axis=0)
        norm_t = np.linalg.norm(tangents, axis=1, keepdims=True)
        norm_t = np.maximum(norm_t, 1e-9)
        unit_tangents = tangents / norm_t

        c_angles = np.unwrap(np.arctan2(unit_tangents[:, 1], unit_tangents[:, 0]))

        # Curvature kappa = ||d(unit_tangent) / ds||
        d_unit = np.gradient(unit_tangents, s, axis=0)
        curvature = np.linalg.norm(d_unit, axis=1)
        return s, c_angles, curvature

    def compute_speed_profile(
        self, s: np.ndarray, curvature: np.ndarray
    ) -> np.ndarray:
        """Compute look-ahead speed profile bounded by curvature, acceleration, and deceleration."""
        n = len(s)
        if n < 2:
            return np.zeros(n)

        ds = np.diff(s)
        # Lateral acceleration limit: v_lat = sqrt(a_lat / curvature)
        speed = np.minimum(
            self.max_speed,
            np.sqrt(self.lateral_acceleration / np.maximum(curvature, 1e-6)),
        )
        # Boundaries have zero feed speed for smooth plunge / retract
        speed[0] = 0.0
        speed[-1] = 0.0

        # Backward pass: deceleration towards sharp corners and endpoint
        for i in range(n - 2, -1, -1):
            max_v = math.sqrt(speed[i + 1] ** 2 + 2.0 * self.acceleration * ds[i])
            speed[i] = min(speed[i], max_v)

        # Forward pass: acceleration from start
        for i in range(1, n):
            max_v = math.sqrt(speed[i - 1] ** 2 + 2.0 * self.acceleration * ds[i - 1])
            speed[i] = min(speed[i], max_v)

        return speed

    def plan_single_cut_segment(
        self,
        points_xy: Sequence[Sequence[float]] | np.ndarray,
        start_time_s: float = 0.0,
        start_step: int = 0,
        initial_c_rad: float | None = None,
    ) -> list[ToolheadCommand]:
        """Plan one continuous cut segment including pre-orient, plunge, cut, and retract."""
        pts = np.asarray(points_xy, dtype=float)
        if len(pts) < 2:
            return []

        s, c_angles, curvature = self.compute_tangent_and_curvature(pts)
        speeds = self.compute_speed_profile(s, curvature)

        commands = []
        curr_time = float(start_time_s)
        curr_step = int(start_step)

        # 1. Pre-orientation in the air at Z_retract
        target_initial_c = float(c_angles[0])
        first_pt = pts[0]
        if initial_c_rad is not None and abs(initial_c_rad - target_initial_c) > 1e-4:
            c_diff = abs(initial_c_rad - target_initial_c)
            # Assume 3.0 rad/s orientation speed
            orient_dt = max(0.05, c_diff / 3.0)
            commands.append(
                ToolheadCommand(
                    step_index=curr_step,
                    timestamp_s=curr_time,
                    x_m=float(first_pt[0]),
                    y_m=float(first_pt[1]),
                    z_m=self.z_retract_m,
                    c_rad=target_initial_c,
                    feed_speed_m_s=0.0,
                    knife_state=KnifeState.RETRACTED,
                    curvature_m_inv=float(curvature[0]),
                    execution_authorized=False,
                )
            )
            curr_step += 1
            curr_time += orient_dt

        # 2. Plunge knife down into fabric
        plunge_dt = abs(self.z_retract_m - self.z_cut_m) / self.z_plunge_speed
        commands.append(
            ToolheadCommand(
                step_index=curr_step,
                timestamp_s=curr_time,
                x_m=float(first_pt[0]),
                y_m=float(first_pt[1]),
                z_m=self.z_retract_m,
                c_rad=target_initial_c,
                feed_speed_m_s=0.0,
                knife_state=KnifeState.PLUNGING,
                curvature_m_inv=float(curvature[0]),
                execution_authorized=False,
            )
        )
        curr_step += 1
        curr_time += plunge_dt

        # 3. Cut along path
        for i in range(len(pts)):
            cmd = ToolheadCommand(
                step_index=curr_step,
                timestamp_s=curr_time,
                x_m=float(pts[i][0]),
                y_m=float(pts[i][1]),
                z_m=self.z_cut_m,
                c_rad=float(c_angles[i]),
                feed_speed_m_s=float(speeds[i]),
                knife_state=KnifeState.CUTTING,
                curvature_m_inv=float(curvature[i]),
                execution_authorized=False,
            )
            commands.append(cmd)
            curr_step += 1
            if i < len(pts) - 1:
                ds = s[i + 1] - s[i]
                avg_speed = max(0.005, (speeds[i] + speeds[i + 1]) / 2.0)
                curr_time += ds / avg_speed

        # 4. Retract knife up out of fabric at last point
        last_pt = pts[-1]
        last_c = float(c_angles[-1])
        retract_dt = abs(self.z_retract_m - self.z_cut_m) / self.z_retract_speed
        commands.append(
            ToolheadCommand(
                step_index=curr_step,
                timestamp_s=curr_time,
                x_m=float(last_pt[0]),
                y_m=float(last_pt[1]),
                z_m=self.z_cut_m,
                c_rad=last_c,
                feed_speed_m_s=0.0,
                knife_state=KnifeState.RETRACTING,
                curvature_m_inv=float(curvature[-1]),
                execution_authorized=False,
            )
        )
        curr_step += 1
        curr_time += retract_dt

        commands.append(
            ToolheadCommand(
                step_index=curr_step,
                timestamp_s=curr_time,
                x_m=float(last_pt[0]),
                y_m=float(last_pt[1]),
                z_m=self.z_retract_m,
                c_rad=last_c,
                feed_speed_m_s=0.0,
                knife_state=KnifeState.RETRACTED,
                curvature_m_inv=float(curvature[-1]),
                execution_authorized=False,
            )
        )
        return commands

    def plan_multi_segments(
        self, segments: Sequence[Sequence[Sequence[float]] | np.ndarray]
    ) -> ToolheadTrajectoryResult:
        """Plan multiple isolated cut segments separated by air traverses (gap handling)."""
        all_commands = []
        curr_time = 0.0
        curr_step = 0
        total_cut_dist = 0.0
        total_trav_dist = 0.0
        diagnostics = []
        last_c = None

        valid_segments = [np.asarray(seg, dtype=float) for seg in segments if len(seg) >= 2]
        if not valid_segments:
            return ToolheadTrajectoryResult(
                commands=(),
                total_distance_m=0.0,
                cut_distance_m=0.0,
                traverse_distance_m=0.0,
                total_time_s=0.0,
                cut_count=0,
                max_speed_m_s=self.max_speed,
                z_cut_m=self.z_cut_m,
                z_retract_m=self.z_retract_m,
                diagnostics=("NO_VALID_SEGMENTS",),
                execution_authorized=False,
            )

        for seg_idx, seg_pts in enumerate(valid_segments):
            # If not first segment, add traverse from previous endpoint to this startpoint
            if all_commands:
                prev_cmd = all_commands[-1]
                trav_dx = seg_pts[0][0] - prev_cmd.x_m
                trav_dy = seg_pts[0][1] - prev_cmd.y_m
                trav_dist = math.hypot(trav_dx, trav_dy)

                if trav_dist > 1e-4:
                    total_trav_dist += trav_dist
                    trav_speed = self.max_speed
                    trav_dt = trav_dist / trav_speed
                    curr_time += trav_dt
                    curr_step += 1
                    # Traverse waypoint with knife fully retracted
                    all_commands.append(
                        ToolheadCommand(
                            step_index=curr_step,
                            timestamp_s=curr_time,
                            x_m=float(seg_pts[0][0]),
                            y_m=float(seg_pts[0][1]),
                            z_m=self.z_retract_m,
                            c_rad=prev_cmd.c_rad,
                            feed_speed_m_s=trav_speed,
                            knife_state=KnifeState.TRAVERSING,
                            curvature_m_inv=0.0,
                            execution_authorized=False,
                        )
                    )
                    curr_step += 1

            seg_diffs = np.diff(seg_pts, axis=0)
            total_cut_dist += float(np.sum(np.linalg.norm(seg_diffs, axis=1)))

            seg_cmds = self.plan_single_cut_segment(
                seg_pts,
                start_time_s=curr_time,
                start_step=curr_step,
                initial_c_rad=last_c,
            )
            if seg_cmds:
                all_commands.extend(seg_cmds)
                curr_time = seg_cmds[-1].timestamp_s
                curr_step = seg_cmds[-1].step_index + 1
                last_c = seg_cmds[-1].c_rad

        return ToolheadTrajectoryResult(
            commands=tuple(all_commands),
            total_distance_m=total_cut_dist + total_trav_dist,
            cut_distance_m=total_cut_dist,
            traverse_distance_m=total_trav_dist,
            total_time_s=curr_time,
            cut_count=len(valid_segments),
            max_speed_m_s=self.max_speed,
            z_cut_m=self.z_cut_m,
            z_retract_m=self.z_retract_m,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            execution_authorized=False,
        )
