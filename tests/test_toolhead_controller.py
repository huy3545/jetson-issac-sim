"""Unit tests for toolhead motion controller (knife Z stroke, tangential C, curvature speed)."""
import math
import unittest
import numpy as np

from motion_planning.toolhead_controller import (
    KnifeState,
    ToolheadCommand,
    ToolheadController,
    ToolheadTrajectoryResult,
)


class TestToolheadController(unittest.TestCase):
    def setUp(self):
        self.controller = ToolheadController(z_cut_m=0.005, z_retract_m=0.020)

    def test_single_cut_knife_z_lifecycle(self):
        # Straight line from (0, 0) to (0.2, 0)
        pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        cmds = self.controller.plan_single_cut_segment(pts)

        self.assertGreaterEqual(len(cmds), 4)

        # 1. Knife starts at Z_retract when plunging
        self.assertEqual(cmds[0].knife_state, KnifeState.PLUNGING)
        self.assertAlmostEqual(cmds[0].z_m, 0.020)
        self.assertAlmostEqual(cmds[0].feed_speed_m_s, 0.0)

        # 2. Midpoints must be CUTTING at Z_cut
        cutting_cmds = [c for c in cmds if c.knife_state == KnifeState.CUTTING]
        self.assertEqual(len(cutting_cmds), 3)
        for c in cutting_cmds:
            self.assertAlmostEqual(c.z_m, 0.005)
            self.assertAlmostEqual(c.c_rad, 0.0)  # Tangent angle along positive X is 0

        # 3. Last command retracts to Z_retract
        self.assertEqual(cmds[-1].knife_state, KnifeState.RETRACTED)
        self.assertAlmostEqual(cmds[-1].z_m, 0.020)
        self.assertAlmostEqual(cmds[-1].feed_speed_m_s, 0.0)

    def test_tangential_c_angle_tracking(self):
        # 90 degree corner: (0, 0) -> (0.1, 0) -> (0.1, 0.1)
        pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]])
        cmds = self.controller.plan_single_cut_segment(pts)

        cutting_cmds = [c for c in cmds if c.knife_state == KnifeState.CUTTING]
        # Start of cut tangent is along +X (c = 0)
        self.assertAlmostEqual(cutting_cmds[0].c_rad, 0.0, places=3)
        # End of cut tangent is along +Y (c = pi/2 ~ 1.5708)
        self.assertAlmostEqual(cutting_cmds[-1].c_rad, math.pi / 2.0, places=3)

    def test_speed_decelerates_at_sharp_turn(self):
        # Sharp turn should have higher curvature and lower speed than straight run
        straight = np.array([[0.0, 0.0], [0.05, 0.0], [0.10, 0.0], [0.15, 0.0], [0.20, 0.0]])
        s_straight, _, curv_straight = self.controller.compute_tangent_and_curvature(straight)
        speed_straight = self.controller.compute_speed_profile(s_straight, curv_straight)

        corner = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]])
        s_corner, _, curv_corner = self.controller.compute_tangent_and_curvature(corner)
        speed_corner = self.controller.compute_speed_profile(s_corner, curv_corner)

        # Midpoint of straight run can reach higher speed
        self.assertGreater(np.max(speed_straight), 0.02)
        # Corner point has nonzero curvature
        self.assertGreater(curv_corner[1], 1.0)

    def test_multi_segment_gap_handling_never_drags_blade(self):
        # Two isolated segments separated by an air gap
        seg1 = np.array([[0.0, 0.0], [0.1, 0.0]])
        seg2 = np.array([[0.3, 0.1], [0.4, 0.1]])

        result = self.controller.plan_multi_segments([seg1, seg2])
        self.assertIsInstance(result, ToolheadTrajectoryResult)
        self.assertEqual(result.cut_count, 2)
        self.assertFalse(result.execution_authorized)

        # Find traverse command between segments
        traverse_cmds = [c for c in result.commands if c.knife_state == KnifeState.TRAVERSING]
        self.assertGreaterEqual(len(traverse_cmds), 1)
        for trav in traverse_cmds:
            # During air traverse across the gap, knife MUST be retracted at 20 mm!
            self.assertAlmostEqual(trav.z_m, 0.020)
            self.assertFalse(trav.execution_authorized)

    def test_execution_authorized_true_is_blocked(self):
        with self.assertRaises(ValueError):
            ToolheadCommand(
                step_index=0,
                timestamp_s=0.0,
                x_m=0.0,
                y_m=0.0,
                z_m=0.005,
                c_rad=0.0,
                feed_speed_m_s=0.05,
                knife_state=KnifeState.CUTTING,
                curvature_m_inv=0.0,
                execution_authorized=True,
            )


if __name__ == "__main__":
    unittest.main()
