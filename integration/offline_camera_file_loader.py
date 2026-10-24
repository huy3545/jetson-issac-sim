"""Offline Camera File Loader and Report Exporter V1.3.

Ingests recorded camera sessions from disk:
    <session_dir>/
        ├── manifest.json
        ├── frame_0000_rgb.png
        ├── frame_0000_depth.npy
        └── ...

Features:
- Schema validation for manifest.json.
- Zero-Trust calibration verification: rejects unverified VERIFIED claims unless matched against VerifiedCalibrationRegistry.
- ReplaySessionReport JSON export with chained audit events.
"""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from data_source import CameraCalibration
from integration.camera_frame_contract import CameraFrameContract, CameraStreamManifest
from integration.offline_camera_replay import ReplayAuditEvent, ReplaySessionReport


class VerifiedCalibrationRegistry:
    """Registry of independently verified camera calibrations from physical/Isaac Sim models."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None):
        self._entries: dict[str, dict[str, Any]] = entries or {}

    def register(
        self,
        calibration_id: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_to_world: np.ndarray,
    ) -> None:
        """Register a verified calibration entry."""
        transform = np.asarray(camera_to_world, dtype=float)
        self._entries[calibration_id] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "camera_to_world": transform.tolist(),
        }

    def is_verified(
        self,
        calibration_id: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_to_world: np.ndarray,
    ) -> bool:
        """Verify calibration against registered ground truth."""
        if calibration_id not in self._entries:
            return False

        entry = self._entries[calibration_id]
        if not (
            np.isclose(entry["fx"], fx)
            and np.isclose(entry["fy"], fy)
            and np.isclose(entry["cx"], cx)
            and np.isclose(entry["cy"], cy)
        ):
            return False

        reg_transform = np.asarray(entry["camera_to_world"], dtype=float)
        cand_transform = np.asarray(camera_to_world, dtype=float)
        return np.allclose(reg_transform, cand_transform, atol=1e-5)


class OfflineCameraFileLoader:
    """Loads recorded offline camera frame sequences from disk directory."""

    def __init__(self, registry: VerifiedCalibrationRegistry | None = None):
        self.registry = registry or VerifiedCalibrationRegistry()

    def load_session(
        self,
        session_dir: Path | str,
    ) -> tuple[CameraStreamManifest, list[CameraFrameContract]]:
        """Load manifest and frame contracts from session directory."""
        dir_path = Path(session_dir).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Session directory does not exist: {dir_path}")

        manifest_file = dir_path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"manifest.json missing in session directory: {dir_path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        # Validate manifest schema
        required_keys = {"session_id", "stream_id", "frames"}
        if not required_keys.issubset(manifest_data.keys()):
            raise ValueError(f"manifest.json missing required keys: {required_keys - set(manifest_data.keys())}")

        session_id = manifest_data["session_id"]
        stream_id = manifest_data["stream_id"]
        clock_domain = manifest_data.get("clock_domain", "camera_clock")
        source_type = manifest_data.get("source_type", "OFFLINE_ISAAC_EXPORT")

        # Parse and verify calibration with Zero-Trust rule
        cal_data = manifest_data.get("calibration", {})
        calibration = self._parse_and_verify_calibration(cal_data)

        frames_data = manifest_data.get("frames", [])
        contracts: list[CameraFrameContract] = []

        for frame_item in frames_data:
            seq_idx = int(frame_item.get("sequence_index", len(contracts)))
            frame_id = str(frame_item.get("frame_id", f"frame_{seq_idx:04d}"))
            t_rgb = frame_item.get("rgb_timestamp")
            t_depth = frame_item.get("depth_timestamp")
            registered = bool(frame_item.get("rgb_depth_registered", False))
            metadata = dict(frame_item.get("metadata", {}))

            # Record claimed evidence status
            if registered:
                metadata["registration_evidence_status"] = "CLAIMED_UNVERIFIED"

            # Check if calibration claim was degraded
            if cal_data.get("calibration_status") == "VERIFIED" and calibration.calibration_status != "VERIFIED":
                metadata["calibration_warning"] = "UNVERIFIED_CALIBRATION_CLAIM"

            # Load RGB image file
            rgb_rel = frame_item.get("rgb_file")
            rgb_image = None
            if rgb_rel:
                rgb_path = dir_path / rgb_rel
                if rgb_path.exists():
                    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    else:
                        metadata["file_error"] = f"CORRUPT_RGB_FILE: {rgb_rel}"
                else:
                    metadata["file_error"] = f"MISSING_RGB_FILE: {rgb_rel}"
            else:
                metadata["file_error"] = "NO_RGB_FILE_SPECIFIED"

            # Load Depth map file
            depth_rel = frame_item.get("depth_file")
            depth_map = None
            if depth_rel:
                depth_path = dir_path / depth_rel
                if depth_path.exists():
                    try:
                        depth_map = np.load(str(depth_path))
                    except Exception as e:
                        metadata["depth_file_error"] = f"CORRUPT_DEPTH_FILE: {depth_rel} ({e})"
                else:
                    metadata["depth_file_error"] = f"MISSING_DEPTH_FILE: {depth_rel}"

            # Create CameraFrameContract
            contract = CameraFrameContract(
                sequence_index=seq_idx,
                frame_id=frame_id,
                rgb_image=rgb_image,
                depth_map=depth_map,
                rgb_timestamp=t_rgb,
                depth_timestamp=t_depth,
                clock_domain=clock_domain,
                rgb_depth_registered=registered,
                calibration=calibration,
                source_type=source_type,
                metadata=metadata,
            )
            contracts.append(contract)

        manifest = CameraStreamManifest(
            stream_id=stream_id,
            source_type=source_type,
            total_expected_frames=len(contracts),
            clock_domain=clock_domain,
            is_registered_evidence=False,
            calibration_id=calibration.calibration_id,
        )

        return manifest, contracts

    def _parse_and_verify_calibration(self, cal_data: dict[str, Any]) -> CameraCalibration:
        """Parse calibration from manifest and enforce Zero-Trust verification."""
        fx = float(cal_data.get("fx", 450.0))
        fy = float(cal_data.get("fy", 450.0))
        cx = float(cal_data.get("cx", 240.0))
        cy = float(cal_data.get("cy", 180.0))
        c2w = np.asarray(cal_data.get("camera_to_world", np.eye(4)), dtype=float)
        claimed_status = cal_data.get("calibration_status", "SYNTHETIC_TEST_ONLY")
        calibration_id = str(cal_data.get("calibration_id", "manifest-calibration"))

        # Zero-Trust verification: If claimed as VERIFIED, check registry
        final_status = claimed_status
        if claimed_status == "VERIFIED":
            if not self.registry.is_verified(calibration_id, fx, fy, cx, cy, c2w):
                # Degrade claim to UNVERIFIED / SYNTHETIC_TEST_ONLY
                final_status = "SYNTHETIC_TEST_ONLY"

        return CameraCalibration(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_world=c2w,
            calibration_status=final_status,
            calibration_id=calibration_id,
        )


class _ReportJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling numpy types, dataclasses, and custom objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "value"):
            return obj.value
        return super().default(obj)


def export_replay_report_json(report: ReplaySessionReport, output_path: Path | str) -> Path:
    """Export comprehensive ReplaySessionReport with full audit events to JSON file."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "session_id": report.session_id,
        "stream_id": report.stream_id,
        "replay_mode": report.replay_mode,
        "total_frames": report.total_frames,
        "accepted_frames": report.accepted_frames,
        "vision_blocked_frames": report.vision_blocked_frames,
        "safety_blocked_frames": report.safety_blocked_frames,
        "contract_rejected_frames": report.contract_rejected_frames,
        "stream_anomaly_rejected_frames": report.stream_anomaly_rejected_frames,
        "stream_anomalies": list(report.stream_anomalies),
        "frame_anomalies": report.frame_anomalies,
        "execution_authorized_never_true": report.execution_authorized_never_true,
        "audit_chain_valid": report.audit_chain_valid,
        "audit_trail_hashes": list(report.audit_trail_hashes),
        "summary": report.summary,
        "audit_events": [
            {
                "session_id": ev.session_id,
                "sequence_index": ev.sequence_index,
                "frame_id": ev.frame_id,
                "rgb_timestamp": ev.rgb_timestamp,
                "depth_timestamp": ev.depth_timestamp,
                "clock_domain": ev.clock_domain,
                "source_type": ev.source_type,
                "status": ev.status,
                "reason_codes": list(ev.reason_codes),
                "payload_content_hash": ev.payload_content_hash,
                "previous_audit_hash": ev.previous_audit_hash,
                "audit_record_hash": ev.audit_record_hash,
                "diagnostic_only": ev.diagnostic_only,
            }
            for ev in report.audit_events
        ],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, cls=_ReportJSONEncoder)

    return out_file
