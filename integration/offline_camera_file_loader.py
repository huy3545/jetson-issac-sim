"""Offline Camera File Loader and Report Exporter V1.3.1.

Ingests recorded camera sessions from disk:
    <session_dir>/
        ├── manifest.json
        ├── frame_0000_rgb.png
        ├── frame_0000_depth.npy
        └── ...

Features:
- Schema validation for manifest.json with strict type checks (rejects non-boolean rgb_depth_registered, non-numeric parameters).
- Zero-Trust calibration verification: distinguishes SYNTHETIC_TEST_ONLY, REGISTERED, and VERIFIED.
- Path traversal, symlink, and pickle injection protection.
- ReplaySessionReport JSON export with chained audit events and final_session_hash.
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


def _validate_numeric(val: Any, field_name: str, allow_negative: bool = True) -> float:
    """Validate that val is a finite number and strictly not a boolean."""
    if type(val) is bool or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"INVALID_MANIFEST_NUMERIC_TYPE: '{field_name}' must be a finite number, "
            f"got {type(val).__name__}: {val!r}"
        )
    float_val = float(val)
    if not np.isfinite(float_val):
        raise ValueError(f"INVALID_MANIFEST_NUMERIC_TYPE: '{field_name}' must be finite, got {float_val}")
    if not allow_negative and float_val < 0:
        raise ValueError(f"INVALID_MANIFEST_NUMERIC_TYPE: '{field_name}' must be non-negative, got {float_val}")
    return float_val


def _resolve_and_verify_path(session_dir: Path, rel_path: str) -> Path:
    """Resolve and verify path safety against path traversal and symlinks."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("PATH_TRAVERSAL_REJECTED: Path must be a non-empty string")

    p = Path(rel_path)
    if p.is_absolute():
        raise ValueError(f"PATH_TRAVERSAL_REJECTED: Absolute path rejected: '{rel_path}'")

    if ".." in p.parts:
        raise ValueError(f"PATH_TRAVERSAL_REJECTED: Directory traversal '..' rejected: '{rel_path}'")

    resolved_session_dir = session_dir.resolve()
    target_path = session_dir / p

    try:
        resolved_path = target_path.resolve()
    except Exception as e:
        raise ValueError(f"PATH_TRAVERSAL_REJECTED: Path resolution error: {e}")

    try:
        if not resolved_path.is_relative_to(resolved_session_dir):
            raise ValueError(f"PATH_TRAVERSAL_REJECTED: Path escapes session directory: '{rel_path}'")
    except AttributeError:
        try:
            resolved_path.relative_to(resolved_session_dir)
        except ValueError:
            raise ValueError(f"PATH_TRAVERSAL_REJECTED: Path escapes session directory: '{rel_path}'")

    # Check for symlink in target_path or any parent up to session_dir
    curr = target_path
    while curr != session_dir and curr.parent != curr:
        if curr.is_symlink():
            raise ValueError(f"PATH_TRAVERSAL_REJECTED: Symlink detected in path: '{rel_path}'")
        curr = curr.parent

    return resolved_path


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
        verification_evidence_hash: str | None = None,
    ) -> str:
        """Register a calibration entry into the registry.

        If verification_evidence_hash is provided and is a non-empty string,
        the status is 'VERIFIED'. Otherwise, status is 'REGISTERED'.
        Returns the assigned status string.
        """
        transform = np.asarray(camera_to_world, dtype=float)
        evidence = (
            str(verification_evidence_hash).strip()
            if (verification_evidence_hash is not None and isinstance(verification_evidence_hash, str))
            else ""
        )
        status = "VERIFIED" if evidence else "REGISTERED"
        self._entries[calibration_id] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "camera_to_world": transform.tolist(),
            "status": status,
            "verification_evidence_hash": evidence if evidence else None,
        }
        return status

    def get_status(
        self,
        calibration_id: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_to_world: np.ndarray,
    ) -> str | None:
        """Query calibration status if parameters match registered entry."""
        if calibration_id not in self._entries:
            return None

        entry = self._entries[calibration_id]
        if not (
            np.isclose(entry["fx"], fx)
            and np.isclose(entry["fy"], fy)
            and np.isclose(entry["cx"], cx)
            and np.isclose(entry["cy"], cy)
        ):
            return None

        reg_transform = np.asarray(entry["camera_to_world"], dtype=float)
        cand_transform = np.asarray(camera_to_world, dtype=float)
        if not np.allclose(reg_transform, cand_transform, atol=1e-5):
            return None

        return entry.get("status", "REGISTERED")

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
        return self.get_status(calibration_id, fx, fy, cx, cy, camera_to_world) == "VERIFIED"


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
        calibration, cal_warning = self._parse_and_verify_calibration(cal_data)

        frames_data = manifest_data.get("frames", [])
        contracts: list[CameraFrameContract] = []

        for frame_item in frames_data:
            if "sequence_index" in frame_item:
                seq_val = frame_item["sequence_index"]
                if type(seq_val) is bool or not isinstance(seq_val, (int, np.integer)):
                    raise ValueError(
                        f"INVALID_MANIFEST_NUMERIC_TYPE: 'sequence_index' must be integer, "
                        f"got {type(seq_val).__name__}: {seq_val!r}"
                    )
                seq_idx = int(seq_val)
            else:
                seq_idx = len(contracts)

            frame_id = str(frame_item.get("frame_id", f"frame_{seq_idx:04d}"))

            t_rgb = frame_item.get("rgb_timestamp")
            if t_rgb is not None:
                t_rgb = _validate_numeric(t_rgb, "rgb_timestamp", allow_negative=False)

            t_depth = frame_item.get("depth_timestamp")
            if t_depth is not None:
                t_depth = _validate_numeric(t_depth, "depth_timestamp", allow_negative=False)

            # Strict Boolean check for rgb_depth_registered
            if "rgb_depth_registered" in frame_item:
                raw_registered = frame_item["rgb_depth_registered"]
                if type(raw_registered) is not bool:
                    raise ValueError(
                        f"INVALID_MANIFEST_BOOLEAN_TYPE: 'rgb_depth_registered' must be a boolean (True/False), "
                        f"got {type(raw_registered).__name__}: {raw_registered!r}"
                    )
                registered = raw_registered
            else:
                registered = False

            metadata = dict(frame_item.get("metadata", {}))

            # Record claimed evidence status
            if registered:
                metadata["registration_evidence_status"] = "CLAIMED_UNVERIFIED"

            # Check if calibration claim was degraded
            if cal_warning:
                metadata["calibration_warning"] = cal_warning

            # Load RGB image file with path traversal check
            rgb_rel = frame_item.get("rgb_file")
            rgb_image = None
            if rgb_rel:
                rgb_path = _resolve_and_verify_path(dir_path, rgb_rel)
                if rgb_path.exists():
                    if rgb_path.stat().st_size > 50 * 1024 * 1024:
                        metadata["file_error"] = f"FILE_SIZE_EXCEEDED: {rgb_rel}"
                    else:
                        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            if bgr.shape[0] > 4096 or bgr.shape[1] > 4096 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
                                metadata["file_error"] = f"IMAGE_DIMENSIONS_EXCEEDED: {rgb_rel}"
                            else:
                                rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        else:
                            metadata["file_error"] = f"CORRUPT_RGB_FILE: {rgb_rel}"
                else:
                    metadata["file_error"] = f"MISSING_RGB_FILE: {rgb_rel}"
            else:
                metadata["file_error"] = "NO_RGB_FILE_SPECIFIED"

            # Load Depth map file with path traversal & allow_pickle=False check
            depth_rel = frame_item.get("depth_file")
            depth_map = None
            if depth_rel:
                depth_path = _resolve_and_verify_path(dir_path, depth_rel)
                if depth_path.exists():
                    if depth_path.stat().st_size > 50 * 1024 * 1024:
                        metadata["depth_file_error"] = f"FILE_SIZE_EXCEEDED: {depth_rel}"
                    else:
                        try:
                            loaded_depth = np.load(str(depth_path), allow_pickle=False)
                            if not isinstance(loaded_depth, np.ndarray):
                                metadata["depth_file_error"] = f"INVALID_DEPTH_ARRAY: {depth_rel} not an ndarray"
                            elif loaded_depth.ndim != 2:
                                metadata["depth_file_error"] = (
                                    f"INVALID_DEPTH_SHAPE: {depth_rel} must be 2D, got shape {loaded_depth.shape}"
                                )
                            elif not np.issubdtype(loaded_depth.dtype, np.floating):
                                metadata["depth_file_error"] = (
                                    f"INVALID_DEPTH_DTYPE: {depth_rel} must be floating point, got {loaded_depth.dtype}"
                                )
                            elif (
                                loaded_depth.shape[0] > 4096
                                or loaded_depth.shape[1] > 4096
                                or loaded_depth.shape[0] == 0
                                or loaded_depth.shape[1] == 0
                            ):
                                metadata["depth_file_error"] = (
                                    f"INVALID_DEPTH_DIMENSIONS: {depth_rel} exceeds safe limits ({loaded_depth.shape})"
                                )
                            else:
                                depth_map = loaded_depth
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

    def _parse_and_verify_calibration(self, cal_data: dict[str, Any]) -> tuple[CameraCalibration, str | None]:
        """Parse calibration from manifest and enforce Zero-Trust verification.

        Returns (CameraCalibration, warning_or_none).
        """
        fx = _validate_numeric(cal_data.get("fx", 450.0), "fx")
        if fx <= 0:
            raise ValueError(f"INVALID_MANIFEST_NUMERIC_TYPE: 'fx' must be positive, got {fx}")
        fy = _validate_numeric(cal_data.get("fy", 450.0), "fy")
        if fy <= 0:
            raise ValueError(f"INVALID_MANIFEST_NUMERIC_TYPE: 'fy' must be positive, got {fy}")
        cx = _validate_numeric(cal_data.get("cx", 240.0), "cx")
        cy = _validate_numeric(cal_data.get("cy", 180.0), "cy")

        c2w_raw = cal_data.get("camera_to_world", np.eye(4).tolist())
        if not isinstance(c2w_raw, (list, tuple, np.ndarray)):
            raise ValueError("INVALID_MANIFEST_NUMERIC_TYPE: 'camera_to_world' must be a 4x4 matrix")
        c2w_arr = np.asarray(c2w_raw)
        if c2w_arr.shape != (4, 4):
            raise ValueError(
                f"INVALID_MANIFEST_NUMERIC_TYPE: 'camera_to_world' must have shape (4, 4), got {c2w_arr.shape}"
            )
        for r_idx, row in enumerate(c2w_raw):
            if not isinstance(row, (list, tuple, np.ndarray)):
                raise ValueError(f"INVALID_MANIFEST_NUMERIC_TYPE: 'camera_to_world' row {r_idx} must be a list/array")
            for c_idx, elem in enumerate(row):
                _validate_numeric(elem, f"camera_to_world[{r_idx}][{c_idx}]")
        c2w = c2w_arr.astype(float)

        claimed_status = cal_data.get("calibration_status", "SYNTHETIC_TEST_ONLY")
        calibration_id = str(cal_data.get("calibration_id", "manifest-calibration"))

        status_in_registry = self.registry.get_status(calibration_id, fx, fy, cx, cy, c2w)
        warning = None

        if claimed_status == "VERIFIED":
            if status_in_registry == "VERIFIED":
                final_status = "VERIFIED"
            elif status_in_registry == "REGISTERED":
                final_status = "REGISTERED"
                warning = "UNVERIFIED_CALIBRATION_CLAIM"
            else:
                final_status = "SYNTHETIC_TEST_ONLY"
                warning = "UNVERIFIED_CALIBRATION_CLAIM"
        elif claimed_status == "REGISTERED":
            if status_in_registry in ("REGISTERED", "VERIFIED"):
                final_status = "REGISTERED"
            else:
                final_status = "SYNTHETIC_TEST_ONLY"
                warning = "UNVERIFIED_CALIBRATION_CLAIM"
        elif claimed_status == "SYNTHETIC_TEST_ONLY":
            final_status = "SYNTHETIC_TEST_ONLY"
        else:
            final_status = "SYNTHETIC_TEST_ONLY"
            warning = "UNVERIFIED_CALIBRATION_CLAIM"

        cal = CameraCalibration(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_world=c2w,
            calibration_status=final_status,
            calibration_id=calibration_id,
        )
        return cal, warning


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
        "final_session_hash": report.final_session_hash,
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
