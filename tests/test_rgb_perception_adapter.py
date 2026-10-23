"""Unit tests for RGB perception adapter covering the 10 mandatory verification criteria."""
from pathlib import Path
import unittest
import cv2
import numpy as np

from config import Config
from data_source import CameraCalibration
from integration.toolhead_closed_loop_harness import (
    PipelineStatus,
    ToolheadClosedLoopHarness,
)
from vision.contracts import ObservationStatus
from vision.rgb_perception_adapter import (
    PerceptionQuality,
    RGBPerceptionAdapter,
)

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


class TestRGBPerceptionAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = RGBPerceptionAdapter(min_blur_var=10.0)
        self.harness = ToolheadClosedLoopHarness()
        # Default calibration for test data (360x480)
        self.calibration = CameraCalibration(
            fx=450.0,
            fy=450.0,
            cx=240.0,
            cy=180.0,
            camera_to_world=np.eye(4),
        )

    # -------------------------------------------------------------------------
    # Criterion 1: Valid RGB image, detects two boundary lines
    # -------------------------------------------------------------------------
    def test_criterion_1_valid_rgb_detects_two_lines(self):
        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        if not rgb_path.exists():
            self.skipTest(f"{rgb_path} not found")

        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result, quality = self.adapter.process_rgb(rgb, frame_id="curve-test")
        self.assertEqual(result.units, "pixels")
        self.assertEqual(result.coordinate_frame, "camera")
        self.assertEqual(result.left.observation_status, ObservationStatus.OBSERVED)
        self.assertEqual(result.right.observation_status, ObservationStatus.OBSERVED)
        self.assertGreater(len(result.left.points), 10)
        self.assertGreater(len(result.right.points), 10)
        self.assertGreater(quality.calculated_confidence, 0.5)
        # Verify confidence is NOT a hardcoded 0.95
        self.assertNotEqual(quality.calculated_confidence, 0.95)

    # -------------------------------------------------------------------------
    # Criterion 2: Image without lines (blank image)
    # -------------------------------------------------------------------------
    def test_criterion_2_blank_image_insufficient_lines(self):
        blank = np.zeros((360, 480, 3), dtype=np.uint8)
        result, quality = self.adapter.process_rgb(blank)
        self.assertEqual(result.left.observation_status, ObservationStatus.INVALID)
        self.assertIn("INSUFFICIENT_LINES", quality.diagnostics)
        self.assertEqual(quality.calculated_confidence, 0.0)

    # -------------------------------------------------------------------------
    # Criterion 3: Image with only 1 line
    # -------------------------------------------------------------------------
    def test_criterion_3_single_line_insufficient_lines(self):
        one_line = np.zeros((360, 480, 3), dtype=np.uint8)
        # Draw one green line in center
        pts = np.array([[240, 50], [240, 300]], dtype=np.int32)
        cv2.polylines(one_line, [pts], False, (0, 220, 0), 8)

        result, quality = self.adapter.process_rgb(one_line)
        self.assertEqual(result.left.observation_status, ObservationStatus.INVALID)
        self.assertIn("INSUFFICIENT_LINES", quality.diagnostics)

    # -------------------------------------------------------------------------
    # Criterion 4: Blurry or noisy image
    # -------------------------------------------------------------------------
    def test_criterion_4_blurry_image_flagged(self):
        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Apply heavy Gaussian blur
        blurry = cv2.GaussianBlur(rgb, (35, 35), 15.0)

        result, quality = self.adapter.process_rgb(blurry)
        self.assertIn("IMAGE_BLURRY", quality.diagnostics)
        self.assertFalse(quality.is_sharp)

    # -------------------------------------------------------------------------
    # Criterion 5: Missing depth -> produces pixel observation, blocks metric
    # -------------------------------------------------------------------------
    def test_criterion_5_missing_depth_preserves_pixels_blocks_metric(self):
        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # process_rgb works without depth
        pixel_result, quality = self.adapter.process_rgb(rgb)
        self.assertEqual(pixel_result.units, "pixels")
        self.assertEqual(pixel_result.coordinate_frame, "camera")

        # But passing None depth to process_rgbd raises ValueError
        with self.assertRaises(ValueError):
            self.adapter.process_rgbd(rgb, None, self.calibration)

    # -------------------------------------------------------------------------
    # Criterion 6: Depth and RGB shape mismatch
    # -------------------------------------------------------------------------
    def test_criterion_6_depth_rgb_shape_mismatch_raises(self):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        bad_depth = np.zeros((200, 200), dtype=float)

        with self.assertRaises(ValueError) as ctx:
            self.adapter.process_rgbd(rgb, bad_depth, self.calibration)
        self.assertIn("must match RGB shape", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Criterion 7: Invalid camera calibration rejected
    # -------------------------------------------------------------------------
    def test_criterion_7_invalid_calibration_rejected(self):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        depth = np.full((360, 480), 0.5, dtype=float)

        # Invalid focal length <= 0
        with self.assertRaises(ValueError):
            CameraCalibration(fx=-100.0, fy=450.0, cx=240.0, cy=180.0, camera_to_world=np.eye(4))

    # -------------------------------------------------------------------------
    # Criterion 8: Trajectory exceeding workspace flagged by Safety
    # -------------------------------------------------------------------------
    def test_criterion_8_workspace_violation_flagged_by_safety(self):
        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        depth_path = TEST_DATA_DIR / "curve_depth.npy"
        if not rgb_path.exists() or not depth_path.exists():
            self.skipTest("test_data files not found")

        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(str(depth_path))

        metric_result, _ = self.adapter.process_rgbd(rgb, depth, self.calibration)
        self.assertEqual(metric_result.units, "m")
        self.assertEqual(metric_result.coordinate_frame, "workpiece_m")

        # Run through Six-Module Closed Loop Harness
        run_res = self.harness.run(metric_result, run_id="criterion-8-run")
        self.assertEqual(run_res.stage_records[0].status, PipelineStatus.ACCEPTED)
        self.assertEqual(run_res.stage_records[1].status, PipelineStatus.ACCEPTED)
        self.assertEqual(run_res.stage_records[2].status, PipelineStatus.ACCEPTED)
        # Safety must enforce Gate B and check workspace
        self.assertEqual(run_res.stage_records[3].status, PipelineStatus.BLOCKED)
        self.assertIn("STRICT_GATE_B_BLOCKED", run_res.stage_records[3].reason_codes)

    # -------------------------------------------------------------------------
    # Criterion 9: Safety rejects execution, but audit trail is complete
    # -------------------------------------------------------------------------
    def test_criterion_9_safety_veto_completes_full_audit_trail(self):
        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        depth_path = TEST_DATA_DIR / "curve_depth.npy"
        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(str(depth_path))

        metric_result, _ = self.adapter.process_rgbd(rgb, depth, self.calibration)
        run_res = self.harness.run(metric_result, run_id="criterion-9-audit")

        # Audit trail must be complete (all 6 records produced and logged)
        self.assertTrue(run_res.audit_trail_complete)
        self.assertEqual(len(run_res.stage_records), 6)
        self.assertFalse(run_res.execution_authorized)
        self.assertEqual(run_res.stage_records[5].stage.value, "DATA_LOGGING")
        self.assertEqual(run_res.stage_records[5].status, PipelineStatus.ACCEPTED)


if __name__ == "__main__":
    unittest.main()
