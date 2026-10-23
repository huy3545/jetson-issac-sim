"""Unit tests for differential drive kinematics adapter."""
import math
import unittest
import numpy as np

from motion_planning.differential_drive_adapter import (
    DifferentialDriveKinematicsAdapter,
    DifferentialWheelCommand,
)


class TestDifferentialDriveAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = DifferentialDriveKinematicsAdapter()

    def test_pure_straight_inverse_kinematics(self):
        # v = 0.10 m/s, omega = 0 rad/s, R = 0.05 m
        # Expected: omega_l = omega_r = 0.10 / 0.05 = 2.0 rad/s
        omega_l, omega_r = self.adapter.inverse_kinematics(0.10, 0.0)
        self.assertAlmostEqual(omega_l, 2.0, places=6)
        self.assertAlmostEqual(omega_r, 2.0, places=6)

    def test_pure_rotation_inverse_kinematics(self):
        # v = 0.0 m/s, omega = 1.0 rad/s, R = 0.05 m, L = 0.30 m
        # v_right = 1.0 * 0.15 = 0.15 m/s -> omega_r = 0.15 / 0.05 = 3.0 rad/s
        # v_left = -0.15 m/s -> omega_l = -3.0 rad/s
        omega_l, omega_r = self.adapter.inverse_kinematics(0.0, 1.0)
        self.assertAlmostEqual(omega_l, -3.0, places=6)
        self.assertAlmostEqual(omega_r, 3.0, places=6)

    def test_round_trip_forward_inverse(self):
        test_cases = [
            (0.05, 0.2),
            (0.15, -0.4),
            (0.0, 0.0),
            (0.12, 0.8),
        ]
        for v, omega in test_cases:
            omega_l, omega_r = self.adapter.inverse_kinematics(v, omega)
            v_reconstructed, omega_reconstructed = self.adapter.forward_kinematics(omega_l, omega_r)
            self.assertAlmostEqual(v, v_reconstructed, places=6)
            self.assertAlmostEqual(omega, omega_reconstructed, places=6)

    def test_convert_straight_waypoints(self):
        waypoints = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        result = self.adapter.convert_waypoints(waypoints, is_cutting=True)

        self.assertEqual(len(result.commands), 4)
        self.assertFalse(result.has_speed_violation)
        self.assertFalse(result.execution_authorized)

        for cmd in result.commands:
            self.assertTrue(cmd.is_cutting)
            self.assertAlmostEqual(cmd.knife_z_m, self.adapter.cut_depth_z)
            self.assertAlmostEqual(cmd.knife_c_rad, 0.0)
            self.assertFalse(cmd.execution_authorized)

    def test_convert_waypoints_retracted_knife(self):
        waypoints = np.array([[0.0, 0.0], [0.1, 0.1]])
        result = self.adapter.convert_waypoints(waypoints, is_cutting=False)
        self.assertEqual(len(result.commands), 2)
        for cmd in result.commands:
            self.assertFalse(cmd.is_cutting)
            self.assertAlmostEqual(cmd.knife_z_m, self.adapter.retract_height_z)

    def test_insufficient_waypoints(self):
        result = self.adapter.convert_waypoints([[0.0, 0.0]])
        self.assertEqual(len(result.commands), 0)
        self.assertIn("INSUFFICIENT_WAYPOINTS", result.diagnostics)

    def test_execution_authorized_is_strictly_forbidden(self):
        with self.assertRaises(ValueError):
            DifferentialWheelCommand(
                step_index=0,
                timestamp_s=0.0,
                v_linear_m_s=0.1,
                omega_rad_s=0.0,
                left_wheel_rad_s=2.0,
                right_wheel_rad_s=2.0,
                knife_z_m=0.005,
                knife_c_rad=0.0,
                is_cutting=True,
                execution_authorized=True,
            )


if __name__ == "__main__":
    unittest.main()
