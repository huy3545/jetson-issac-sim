"""Differential drive kinematics adapter for fabric feed cutting systems.

Converts Cartesian trajectory waypoints (linear velocity v, angular velocity omega)
into differential drive wheel angular velocities (omega_left, omega_right) for
two-wheel feed machines such as scenes/fabric_feed_cutting_world.usda.

This module is non-executable by default (execution_authorized=False) and purely
provides kinematic transformations and limits validation.
"""
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from simulation.isaac_manifest import IsaacSceneManifest, load_manifest


@dataclass(frozen=True)
class DifferentialWheelCommand:
    """Read-only kinematic setpoint for two-wheel fabric feeder and knife."""
    step_index: int
    timestamp_s: float
    v_linear_m_s: float
    omega_rad_s: float
    left_wheel_rad_s: float
    right_wheel_rad_s: float
    knife_z_m: float
    knife_c_rad: float
    is_cutting: bool
    execution_authorized: bool = False

    def __post_init__(self):
        if self.execution_authorized:
            raise ValueError("Execution cannot be authorized at kinematic adapter stage")


@dataclass(frozen=True)
class DifferentialTrajectoryResult:
    commands: tuple[DifferentialWheelCommand, ...]
    wheel_radius_m: float
    wheel_track_width_m: float
    max_feed_speed_m_s: float
    max_angular_velocity_rad_s: float
    has_speed_violation: bool
    diagnostics: tuple[str, ...]
    execution_authorized: bool = False


class DifferentialDriveKinematicsAdapter:
    """Transforms continuous trajectory into differential feed wheel setpoints."""

    def __init__(self, manifest: IsaacSceneManifest | None = None):
        self.manifest = manifest or load_manifest()
        kinematics = self.manifest.kinematics
        self.wheel_radius_m = float(kinematics["wheel_radius_m"])
        self.wheel_track_width_m = float(kinematics["wheel_track_width_m"])
        self.max_feed_speed = float(kinematics["max_feed_speed_m_s"])
        self.max_angular_vel = float(kinematics["max_angular_velocity_rad_s"])
        self.cut_depth_z = float(kinematics["knife_cut_depth_z_m"])
        self.retract_height_z = float(kinematics["knife_retract_height_z_m"])

    def inverse_kinematics(self, v_m_s: float, omega_rad_s: float) -> tuple[float, float]:
        """Compute (omega_left, omega_right) in rad/s from linear and angular velocity."""
        half_width = self.wheel_track_width_m / 2.0
        v_right = v_m_s + (omega_rad_s * half_width)
        v_left = v_m_s - (omega_rad_s * half_width)

        omega_r = v_right / self.wheel_radius_m
        omega_l = v_left / self.wheel_radius_m
        return omega_l, omega_r

    def forward_kinematics(self, omega_left_rad_s: float, omega_right_rad_s: float) -> tuple[float, float]:
        """Compute (v_m_s, omega_rad_s) from wheel angular velocities."""
        v_l = omega_left_rad_s * self.wheel_radius_m
        v_r = omega_right_rad_s * self.wheel_radius_m

        v_linear = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.wheel_track_width_m
        return v_linear, omega

    def convert_waypoints(
        self,
        waypoints_xy: Sequence[Sequence[float]] | np.ndarray,
        speeds_m_s: Sequence[float] | None = None,
        timestamps_s: Sequence[float] | None = None,
        is_cutting: bool = True,
    ) -> DifferentialTrajectoryResult:
        """Convert path points into differential wheel command sequence."""
        pts = np.asarray(waypoints_xy, dtype=float)
        if len(pts) < 2:
            return DifferentialTrajectoryResult(
                commands=(),
                wheel_radius_m=self.wheel_radius_m,
                wheel_track_width_m=self.wheel_track_width_m,
                max_feed_speed_m_s=self.max_feed_speed,
                max_angular_velocity_rad_s=self.max_angular_vel,
                has_speed_violation=False,
                diagnostics=("INSUFFICIENT_WAYPOINTS",),
                execution_authorized=False,
            )

        n = len(pts)
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        # Pad last angle to match points length
        c_angles = np.append(angles, angles[-1])

        default_speed = min(0.10, self.max_feed_speed)
        v_profile = np.full(n, default_speed) if speeds_m_s is None else np.asarray(speeds_m_s, dtype=float)

        if timestamps_s is None:
            # Generate synthetic timestamps based on distance and speed
            dt_segment = np.where(v_profile[:-1] > 1e-6, dists / v_profile[:-1], 0.1)
            time_seq = np.insert(np.cumsum(dt_segment), 0, 0.0)
        else:
            time_seq = np.asarray(timestamps_s, dtype=float)

        commands = []
        violations = False
        diagnostics = []

        knife_z = self.cut_depth_z if is_cutting else self.retract_height_z

        for i in range(n):
            v = float(v_profile[i])
            # Angular velocity from angle difference over time
            if i < n - 1:
                dt = max(1e-4, float(time_seq[i + 1] - time_seq[i]))
                d_theta = float(angles[i] - (angles[i - 1] if i > 0 else angles[0]))
                # Normalize angle difference to [-pi, pi]
                d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                omega = d_theta / dt
            else:
                omega = 0.0

            if abs(v) > self.max_feed_speed:
                violations = True
                diagnostics.append(f"SPEED_LIMIT_EXCEEDED_STEP_{i}")

            if abs(omega) > self.max_angular_vel:
                violations = True
                diagnostics.append(f"ANGULAR_VEL_EXCEEDED_STEP_{i}")

            omega_l, omega_r = self.inverse_kinematics(v, omega)

            cmd = DifferentialWheelCommand(
                step_index=i,
                timestamp_s=float(time_seq[i]),
                v_linear_m_s=v,
                omega_rad_s=omega,
                left_wheel_rad_s=omega_l,
                right_wheel_rad_s=omega_r,
                knife_z_m=knife_z,
                knife_c_rad=float(c_angles[i]),
                is_cutting=is_cutting,
                execution_authorized=False,
            )
            commands.append(cmd)

        return DifferentialTrajectoryResult(
            commands=tuple(commands),
            wheel_radius_m=self.wheel_radius_m,
            wheel_track_width_m=self.wheel_track_width_m,
            max_feed_speed_m_s=self.max_feed_speed,
            max_angular_velocity_rad_s=self.max_angular_vel,
            has_speed_violation=violations,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            execution_authorized=False,
        )
