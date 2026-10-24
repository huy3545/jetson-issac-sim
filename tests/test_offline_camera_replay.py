"""Unit tests for Offline Camera Replay Bridge V1.3.

Verifies:
1. Valid synthetic stream replays through 6 modules with dry-run labeling and chained audit.
2. Missing timestamp evidence blocks metric projection and records valid audit event.
3. Unregistered RGB-D frame is blocked.
4. Synthetic calibration rejected when allow_synthetic_dry_run=False.
5. STRICT mode blocks stream anomalies (STREAM_ANOMALY_REJECTED) while DIAGNOSTIC mode captures telemetry.
6. Error recovery across subsequent frames without corrupting audit trail.
7. Audit chain verification (verify_audit_chain) detects tampering and deletions.
8. OfflineCameraFileLoader loads session from disk, enforces Zero-Trust calibration, and exports JSON report.
"""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from data_source import CameraCalibration
from integration.camera_frame_contract import CameraFrameContract, validate_frame_contract
from integration.offline_camera_file_loader import (
    OfflineCameraFileLoader,
    VerifiedCalibrationRegistry,
    export_replay_report_json,
)
from integration.offline_camera_replay import (
    OfflineCameraReplay,
    ReplayAuditEvent,
    ReplaySessionReport,
    verify_audit_chain,
)

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


class TestOfflineCameraReplay(unittest.TestCase):
    def setUp(self):
        self.replay = OfflineCameraReplay(allow_synthetic_dry_run=True, replay_mode="STRICT")
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
        self.assertEqual(report.stream_anomaly_rejected_frames, 0)

        # Gate B must never authorize execution
        self.assertTrue(report.execution_authorized_never_true)
        # All 3 frames produced audit hashes and audit events
        self.assertEqual(len(report.audit_trail_hashes), 3)
        self.assertEqual(len(report.audit_events), 3)

        # Cryptographic chain must be valid
        self.assertTrue(report.audit_chain_valid)
        self.assertTrue(verify_audit_chain(report))

        # Verify dry-run synthetic anomaly tag is present in anomalies
        self.assertIn("CALIBRATION_SYNTHETIC_TEST_ONLY", report.stream_anomalies)

    # -------------------------------------------------------------------------
    # Test 2: Missing timestamp evidence blocks metric projection
    # -------------------------------------------------------------------------
    def test_2_missing_timestamp_evidence_blocks_metric_projection(self):
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
        # Audit event must be properly populated
        self.assertIsNotNone(res_1.content_hash)
        self.assertEqual(res_1.audit_event.status, "CONTRACT_REJECTED")

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
        unreg_frame = self._make_frame(0, 0.000, 0.000, registered=False)
        res = self.replay.replay_frame(unreg_frame)
        self.assertEqual(res.status, "CONTRACT_REJECTED")
        self.assertIn("UNREGISTERED_RGBD", res.anomalies)
        self.assertFalse(res.execution_authorized)
        self.assertIsNotNone(res.content_hash)

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
        self.assertIsNotNone(res.content_hash)

    # -------------------------------------------------------------------------
    # Test 5: STRICT vs DIAGNOSTIC mode for stream anomalies
    # -------------------------------------------------------------------------
    def test_5_strict_vs_diagnostic_mode_for_stream_anomalies(self):
        frames = [
            self._make_frame(0, 0.000, 0.000),
            # Gap: jumps from 0 to 2 (missing 1)
            self._make_frame(2, 0.033, 0.033),
            # Non-monotonic: timestamp goes backward
            self._make_frame(3, 0.020, 0.020),
        ]

        # 1. STRICT Mode: Rejects processing on stream anomaly
        replay_strict = OfflineCameraReplay(allow_synthetic_dry_run=True, replay_mode="STRICT")
        report_strict = replay_strict.replay_stream(frames, session_id="test-strict")
        self.assertEqual(report_strict.total_frames, 3)
        self.assertEqual(report_strict.accepted_frames, 1)
        self.assertEqual(report_strict.stream_anomaly_rejected_frames, 2)
        self.assertTrue(report_strict.audit_chain_valid)

        # 2. DIAGNOSTIC Mode: Processes frames for telemetry, tagged diagnostic_only
        replay_diag = OfflineCameraReplay(allow_synthetic_dry_run=True, replay_mode="DIAGNOSTIC")
        report_diag = replay_diag.replay_stream(frames, session_id="test-diag")
        self.assertEqual(report_diag.total_frames, 3)
        self.assertEqual(report_diag.accepted_frames, 3)
        self.assertEqual(report_diag.stream_anomaly_rejected_frames, 0)
        self.assertTrue(report_diag.audit_chain_valid)

        # Check diagnostic_only flag in audit events for frames 2 and 3
        self.assertFalse(report_diag.audit_events[0].diagnostic_only)
        self.assertTrue(report_diag.audit_events[1].diagnostic_only)
        self.assertTrue(report_diag.audit_events[2].diagnostic_only)

    # -------------------------------------------------------------------------
    # Test 6: Error recovery across subsequent frames without corrupting audit trail
    # -------------------------------------------------------------------------
    def test_6_error_recovery_maintains_subsequent_frame_processing(self):
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

        # All 3 frames have cryptographic audit hashes linked in order
        self.assertEqual(len(report.audit_events), 3)
        self.assertTrue(report.audit_chain_valid)
        self.assertTrue(verify_audit_chain(report))
        self.assertTrue(report.execution_authorized_never_true)

    # -------------------------------------------------------------------------
    # Test 7: Audit chain verification detects tampering and deletions
    # -------------------------------------------------------------------------
    def test_7_audit_chain_tamper_detection(self):
        frames = [
            self._make_frame(0, 0.000, 0.000),
            self._make_frame(1, 0.033, 0.033),
            self._make_frame(2, 0.066, 0.066),
        ]
        report = self.replay.replay_stream(frames, session_id="test-tamper")
        self.assertTrue(verify_audit_chain(report))

        # 1. Tamper with an event's status
        tampered_events = list(report.audit_events)
        original_ev = tampered_events[1]
        tampered_ev = ReplayAuditEvent(
            session_id=original_ev.session_id,
            sequence_index=original_ev.sequence_index,
            frame_id=original_ev.frame_id,
            rgb_timestamp=original_ev.rgb_timestamp,
            depth_timestamp=original_ev.depth_timestamp,
            clock_domain=original_ev.clock_domain,
            source_type=original_ev.source_type,
            status="TAMPERED_STATUS",
            reason_codes=original_ev.reason_codes,
            payload_content_hash=original_ev.payload_content_hash,
            previous_audit_hash=original_ev.previous_audit_hash,
            audit_record_hash=original_ev.audit_record_hash,
            diagnostic_only=original_ev.diagnostic_only,
        )
        tampered_events[1] = tampered_ev
        tampered_report = ReplaySessionReport(
            session_id=report.session_id,
            stream_id=report.stream_id,
            replay_mode=report.replay_mode,
            total_frames=report.total_frames,
            accepted_frames=report.accepted_frames,
            vision_blocked_frames=report.vision_blocked_frames,
            safety_blocked_frames=report.safety_blocked_frames,
            contract_rejected_frames=report.contract_rejected_frames,
            stream_anomaly_rejected_frames=report.stream_anomaly_rejected_frames,
            audit_events=tuple(tampered_events),
        )
        self.assertFalse(verify_audit_chain(tampered_report))

        # 2. Delete an event from the middle of the chain
        truncated_events = (report.audit_events[0], report.audit_events[2])
        truncated_report = ReplaySessionReport(
            session_id=report.session_id,
            stream_id=report.stream_id,
            replay_mode=report.replay_mode,
            total_frames=report.total_frames,
            accepted_frames=report.accepted_frames,
            vision_blocked_frames=report.vision_blocked_frames,
            safety_blocked_frames=report.safety_blocked_frames,
            contract_rejected_frames=report.contract_rejected_frames,
            stream_anomaly_rejected_frames=report.stream_anomaly_rejected_frames,
            audit_events=truncated_events,
        )
        self.assertFalse(verify_audit_chain(truncated_report))

    # -------------------------------------------------------------------------
    # Test 8: OfflineCameraFileLoader loads session from disk and exports JSON
    # -------------------------------------------------------------------------
    def test_8_file_loader_and_json_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            # Save two dummy frames
            cv2.imwrite(str(tmp_dir / "frame_0000_rgb.png"), cv2.cvtColor(self.sample_rgb, cv2.COLOR_RGB2BGR))
            np.save(str(tmp_dir / "frame_0000_depth.npy"), self.sample_depth)
            cv2.imwrite(str(tmp_dir / "frame_0001_rgb.png"), cv2.cvtColor(self.sample_rgb, cv2.COLOR_RGB2BGR))
            np.save(str(tmp_dir / "frame_0001_depth.npy"), self.sample_depth)

            manifest_content = {
                "session_id": "disk-test-session",
                "stream_id": "disk-camera-01",
                "clock_domain": "camera_clock",
                "source_type": "OFFLINE_ISAAC_EXPORT",
                "calibration": {
                    "fx": 450.0,
                    "fy": 450.0,
                    "cx": 240.0,
                    "cy": 180.0,
                    "camera_to_world": np.eye(4).tolist(),
                    "calibration_status": "SYNTHETIC_TEST_ONLY",
                    "calibration_id": "disk-cal-01",
                },
                "frames": [
                    {
                        "sequence_index": 0,
                        "frame_id": "frame_0000",
                        "rgb_file": "frame_0000_rgb.png",
                        "depth_file": "frame_0000_depth.npy",
                        "rgb_timestamp": 0.0,
                        "depth_timestamp": 0.0,
                        "rgb_depth_registered": True,
                    },
                    {
                        "sequence_index": 1,
                        "frame_id": "frame_0001",
                        "rgb_file": "frame_0001_rgb.png",
                        "depth_file": "frame_0001_depth.npy",
                        "rgb_timestamp": 0.033,
                        "depth_timestamp": 0.033,
                        "rgb_depth_registered": True,
                    },
                ],
            }
            with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_content, f, indent=2)

            loader = OfflineCameraFileLoader()
            manifest, contracts = loader.load_session(tmp_dir)

            self.assertEqual(len(contracts), 2)
            self.assertEqual(manifest.stream_id, "disk-camera-01")
            self.assertIsNotNone(contracts[0].rgb_image)
            self.assertIsNotNone(contracts[0].depth_map)

            # Replay the loaded stream
            replay = OfflineCameraReplay(allow_synthetic_dry_run=True)
            report = replay.replay_stream(contracts, session_id="disk-session")
            self.assertEqual(report.total_frames, 2)
            self.assertEqual(report.accepted_frames, 2)
            self.assertTrue(report.audit_chain_valid)

            # Export to JSON
            out_json = tmp_dir / "replay_report.json"
            export_replay_report_json(report, out_json)
            self.assertTrue(out_json.exists())

            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["session_id"], "disk-session")
            self.assertEqual(data["total_frames"], 2)
            self.assertTrue(data["audit_chain_valid"])
            self.assertEqual(len(data["audit_events"]), 2)

    # -------------------------------------------------------------------------
    # Test 9: Zero-Trust calibration blocks unverified VERIFIED claims
    # -------------------------------------------------------------------------
    def test_9_zero_trust_calibration_blocks_unverified_claim(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            cv2.imwrite(str(tmp_dir / "frame_0000_rgb.png"), cv2.cvtColor(self.sample_rgb, cv2.COLOR_RGB2BGR))
            np.save(str(tmp_dir / "frame_0000_depth.npy"), self.sample_depth)

            # Manifest falsely claims "VERIFIED" without being registered
            manifest_content = {
                "session_id": "zero-trust-session",
                "stream_id": "disk-camera-01",
                "calibration": {
                    "fx": 450.0,
                    "fy": 450.0,
                    "cx": 240.0,
                    "cy": 180.0,
                    "camera_to_world": np.eye(4).tolist(),
                    "calibration_status": "VERIFIED",
                    "calibration_id": "unregistered-cal-id",
                },
                "frames": [
                    {
                        "sequence_index": 0,
                        "frame_id": "frame_0000",
                        "rgb_file": "frame_0000_rgb.png",
                        "depth_file": "frame_0000_depth.npy",
                        "rgb_timestamp": 0.0,
                        "depth_timestamp": 0.0,
                        "rgb_depth_registered": True,
                    }
                ],
            }
            with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_content, f, indent=2)

            # 1. Unregistered -> degraded to SYNTHETIC_TEST_ONLY with warning
            loader_empty = OfflineCameraFileLoader(registry=VerifiedCalibrationRegistry())
            _, contracts_unreg = loader_empty.load_session(tmp_dir)
            self.assertEqual(contracts_unreg[0].calibration.calibration_status, "SYNTHETIC_TEST_ONLY")
            self.assertEqual(
                contracts_unreg[0].metadata.get("calibration_warning"),
                "UNVERIFIED_CALIBRATION_CLAIM",
            )

            # When allow_synthetic_dry_run=False, degraded calibration is blocked
            strict_replay = OfflineCameraReplay(allow_synthetic_dry_run=False)
            res = strict_replay.replay_frame(contracts_unreg[0])
            self.assertEqual(res.status, "VISION_BLOCKED")

            # 2. Officially registered in registry -> accepted as VERIFIED
            registry = VerifiedCalibrationRegistry()
            registry.register(
                calibration_id="unregistered-cal-id",
                fx=450.0,
                fy=450.0,
                cx=240.0,
                cy=180.0,
                camera_to_world=np.eye(4),
            )
            loader_reg = OfflineCameraFileLoader(registry=registry)
            _, contracts_reg = loader_reg.load_session(tmp_dir)
            self.assertEqual(contracts_reg[0].calibration.calibration_status, "VERIFIED")
            self.assertNotIn("calibration_warning", contracts_reg[0].metadata)

            # When allow_synthetic_dry_run=False, legitimately verified calibration passes
            strict_replay.reset_stream_state()
            res_reg = strict_replay.replay_frame(contracts_reg[0])
            self.assertEqual(res_reg.status, "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
