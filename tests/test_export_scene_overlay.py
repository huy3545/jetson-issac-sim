"""Unit tests for export_scene_overlay module."""
import unittest
from simulation.export_scene_overlay import canonical_vision, main
from vision.contracts import VisionResult


class TestExportSceneOverlay(unittest.TestCase):
    def test_canonical_vision(self):
        vision = canonical_vision()
        self.assertIsInstance(vision, VisionResult)
        self.assertEqual(vision.coordinate_frame, "workpiece_m")
        self.assertEqual(vision.units, "m")
        self.assertEqual(len(vision.left.points), 3)
        self.assertEqual(len(vision.right.points), 3)
        self.assertTrue(vision.left.verified)
        self.assertTrue(vision.right.verified)

    def test_cli_without_pxr_exits_gracefully(self):
        # In environment without pxr, main() must return 2 (ISAAC_RUNTIME_NOT_VERIFIED)
        exit_code = main(["--output", "artifacts/preview/test_overlay.usda"])
        # Should be either 2 (no pxr) or 0 (if pxr is available)
        self.assertIn(exit_code, (0, 2))


if __name__ == "__main__":
    unittest.main()
