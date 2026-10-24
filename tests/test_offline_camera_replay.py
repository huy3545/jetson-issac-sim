"""Unit tests for Offline Camera Replay Bridge (Option C).

Verifies 6 mandatory test cases:
1. Valid synthetic stream replays through 6 modules with dry-run labeling.
2. Missing timestamp evidence blocks metric projection.
3. Unregistered RGB-D frame is blocked.
4. Synthetic calibration rejected when allow_synthetic_dry_run=False.
5. Stream gap and non-monotonic timestamp anomalies detected.
6. Error recovery across subsequent frames without corrupting audit trail.
"""
from pathlib import Path
import unittest
import cv2
import numpy as np

from data_source import CameraCalibration
from integration.camera_frame_contract import CameraFrameContract, validate_frame_contract
from integration.offline_camera_replay import OfflineCameraReplay, ReplaySessionReport
from integration.toolhead_closed_loop_harness import ToolheadClosedLoopHarness

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


class TestOfflineCameraReplay(unittest.TestCase):
    def setUp(self):
        self.replay = OfflineCameraReplay(allow_synthetic_dry_run=True)
        self.calibration = CameraCalibration(
            fx=450.0,
            fy=450.0,
            cx=240.0,
            cy=180.0,
            camera_to_world=np.eye(4),
            calibration_status="SYNTHETIC_TEST_ONLY",
            calibration_id="replay-test-cal",
        )

        rgb_path = TEST_DATA_DIR / "curve_rgb.png"
        depth_path = TEST_DATA_DIR / "curve_depth.npy"
        if rgb_path.exists() and depth_path.exists():
            bgr = cv2.imread(str(rgb_path))
            self.sample_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.sample_depth = np.load(str(depth_path))
        else:
            # Fallback synthetic pattern (360x480) with 2 green lines
            self.sample_rgb = np.full((360, 480, 3), 20, dtype=np.uint8)
            cv2.line(self.sample_rgb, (200, 40), (200, 320), (0, 220, 0), 8)
            cv2.line(self.sample_rgb, (280, 40), (280, 320), (0, 220, 0), 8)
            self.sample_depth = np.full((360, 480), 0.5, dtype=float)

    def _make_frame(
        self,
        seq: int,
        t_rgb: float,
        t_depth: float,
        registered: bool = True,
        cal: CameraCalibration | None = None,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> CameraFrameContract:
        return CameraFrameContract(
            sequence_index=seq,
            frame_id=f"frame_{seq:04d}",
            rgb_image=self.sample_rgb if rgb is None else rgb,
            depth_map=self.sample_depth if depth is None else depth,
            rgb_timestamp=t_rgb,
            depth_timestamp=t_depth,
            clock_domain="camera_clock",
            rgb_depth_registered=registered,
            calibration=self.calibration if cal is None else cal,
            source_type="OFFLINE_SYNTHETIC",
        )

    # -------------------------------------------------------------------------
    # Test 1: Valid synthetic stream replays through 6 modules with dry-run labeling
    # -------------------------------------------------------------------------
    def test_1_valid_synthetic_stream_replays_through_six_modules_dry_run(self):
        frames = [
            self._make_frame(0, 0.000, 0.000),
            self._make_frame(1, 0.033, 0.033),
            self._make_frame(2, 0.066, 0.066),
        ]

        report = self.replay.replay_stream(frames, session_id="test-session-1")
        self.assertEqual(report.total_frames, 3)
        self.assertEqual(report.accepted_frames, 3)
        self.assertEqual(report.vision_blocked_frames, 0)
        self.assertEqual(report.safety_blocked_frames, 0)
        self.assertEqual(report.contract_rejected_frames, 0)

        # Gate B must never authorize execution
        self.assertTrue(report.execution_authorized_never_true)
        # All 3 frames produced audit hashes
        self.assertEqual(len(report.audit_trail_hashes), 3)

        # Verify dry-run synthetic anomaly tag is present in anomalies
        self.assertIn("CALIBRATION_SYNTHETIC_TEST_ONLY", report.stream_anomalies)

    # -------------------------------------------------------------------------
    # Test 2: Missing timestamp evidence blocks metric projection
    # -------------------------------------------------------------------------
    def test_2_missing_timestamp_evidence_blocks_metric_projection(self):
        # Frame missing rgb_timestamp
        bad_frame_1 = CameraFrameContract(
            sequence_index=0,
            frame_id="frame_missing_t_rgb",
            rgb_image=self.sample_rgb,
            depth_map=self.sample_depth,
            rgb_timestamp=None,
            depth_timestamp=0.0,
            rgb_depth_registered=True,
            calibration=self.calibration,
        )
        res_1 = self.replay.replay_frame(bad_frame_1)
        self.assertEqual(res_1.status, "CONTRACT_REJECTED")
        self.assertIn("MISSING_RGB_TIMESTAMP", res_1.anomalies)
        self.assertFalse(res_1.execution_authorized)

        # Frame missing depth_timestamp
        bad_frame_2 = CameraFrameContract(
            sequence_index=1,
            frame_id="frame_missing_t_depth",
            rgb_image=self.sample_rgb,
            depth_map=self.sample_depth,
            rgb_timestamp=0.033,
            depth_timestamp=None,
            rgb_depth_registered=True,
            calibration=self.calibration,
        )
        res_2 = self.replay.replay_frame(bad_frame_2)
        self.assertEqual(res_2.status, "CONTRACT_REJECTED")
        self.assertIn("MISSING_DEPTH_TIMESTAMP", res_2.anomalies)

    # -------------------------------------------------------------------------
    # Test 3: Unregistered RGB-D frame is blocked
    # -------------------------------------------------------------------------
    def test_3_unregistered_rgb_depth_rejected(self):
        # Frame with rgb_depth_registered=False
        unreg_frame = self._make_frame(0, 0.000, 0.000, registered=False)
        res = self.replay.replay_frame(unreg_frame)
        self.assertEqual(res.status, "CONTRACT_REJECTED")
        self.assertIn("UNREGISTERED_RGBD", res.anomalies)
        self.assertFalse(res.execution_authorized)

    # -------------------------------------------------------------------------
    # Test 4: Synthetic calibration rejected when allow_synthetic_dry_run=False
    # -------------------------------------------------------------------------
    def test_4_synthetic_calibration_rejected_when_not_dry_run(self):
        strict_replay = OfflineCameraReplay(allow_synthetic_dry_run=False)
        frame = self._make_frame(0, 0.000, 0.000, cal=self.calibration)

        res = strict_replay.replay_frame(frame)
        self.assertEqual(res.status, "VISION_BLOCKED")
        self.assertTrue(any("SYNTHETIC_CALIBRATION_REJECTED" in a for a in res.anomalies))
        self.assertFalse(res.execution_authorized)

    # -------------------------------------------------------------------------
    # Test 5: Stream gap and non-monotonic timestamp anomalies detected
    # -------------------------------------------------------------------------
    def test_5_stream_gap_and_non_monotonic_anomalies_detected(self):
        frames = [
            self._make_frame(0, 0.000, 0.000),
            # Gap: jumps from 0 to 2 (missing 1)
            self._make_frame(2, 0.033, 0.033),
            # Non-monotonic: timestamp goes backward from 0.033 to 0.020
            self._make_frame(3, 0.020, 0.020),
        ]

        report = self.replay.replay_stream(frames, session_id="test-anomalies")
        self.assertEqual(report.total_frames, 3)

        # Gap should be recorded on frame 2
        anomalies_f2 = report.frame_anomalies.get(2, [])
        self.assertTrue(any("FRAME_GAP_DETECTED" in a for a in anomalies_f2))

        # Non-monotonic should be recorded on frame 3
        anomalies_f3 = report.frame_anomalies.get(3, [])
        self.assertTrue(any("NON_MONOTONIC_TIMESTAMP" in a for a in anomalies_f3))

    # -------------------------------------------------------------------------
    # Test 6: Error recovery across subsequent frames without corrupting audit trail
    # -------------------------------------------------------------------------
    def test_6_error_recovery_maintains_subsequent_frame_processing(self):
        # Sequence: Valid(0) -> Corrupted Blank(1) -> Valid(2)
        blank_rgb = np.zeros((360, 480, 3), dtype=np.uint8)

        frames = [
            self._make_frame(0, 0.000, 0.000),
            self._make_frame(1, 0.033, 0.033, rgb=blank_rgb),  # Fails at vision (no lines)
            self._make_frame(2, 0.066, 0.066),  # Recovers cleanly
        ]

        report = self.replay.replay_stream(frames, session_id="test-recovery")
        self.assertEqual(report.total_frames, 3)
        self.assertEqual(report.accepted_frames, 2)
        self.assertEqual(report.vision_blocked_frames, 1)

        # Audit trail must contain hashes from the 2 accepted runs
        self.assertEqual(len(report.audit_trail_hashes), 2)
        self.assertTrue(report.execution_authorized_never_true)


if __name__ == "__main__":
    unittest.main()
