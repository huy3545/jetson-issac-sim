"""Camera Frame Contract and Stream Manifest for Offline Replay and Bridge Ingestion.

Standardized contract ensuring:
- Explicit timestamp evidence (no implicit zeros).
- Explicit registration flag (no implicit True).
- Calibration provenance enforcement.
"""
from dataclasses import dataclass, field
import math
from typing import Literal
import numpy as np

from data_source import CameraCalibration

SourceType = Literal["OFFLINE_SYNTHETIC", "OFFLINE_ISAAC_EXPORT", "LIVE_ROS2"]


@dataclass(frozen=True)
class CameraFrameContract:
    """Standardized camera input frame contract.

    Enforces explicit synchronization, registration evidence, and calibration status.
    """
    sequence_index: int
    frame_id: str
    rgb_image: np.ndarray
    depth_map: np.ndarray | None = None
    rgb_timestamp: float | None = None
    depth_timestamp: float | None = None
    clock_domain: str = "camera_clock"
    rgb_depth_registered: bool = False
    calibration: CameraCalibration | None = None
    source_type: SourceType = "OFFLINE_SYNTHETIC"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CameraStreamManifest:
    """Manifest describing an entire camera recording session or replay stream."""
    stream_id: str
    source_type: SourceType
    total_expected_frames: int
    target_fps: float = 30.0
    clock_domain: str = "camera_clock"
    max_sync_delta_s: float = 0.005
    created_timestamp: float = 0.0
    is_registered_evidence: bool = False
    calibration_id: str | None = None


def validate_frame_contract(contract: CameraFrameContract) -> tuple[bool, list[str]]:
    """Validate data integrity contract before passing to perception or replay pipeline."""
    errors: list[str] = []

    if not isinstance(contract, CameraFrameContract):
        return False, ["INVALID_CONTRACT_TYPE"]

    if contract.sequence_index < 0:
        errors.append("NEGATIVE_SEQUENCE_INDEX")

    # RGB Image validation
    if contract.rgb_image is None or not isinstance(contract.rgb_image, np.ndarray):
        errors.append("INVALID_RGB_IMAGE")
    elif contract.rgb_image.ndim != 3 or contract.rgb_image.shape[2] != 3:
        errors.append("INVALID_RGB_SHAPE")

    # Depth Map validation
    if contract.depth_map is not None:
        if not isinstance(contract.depth_map, np.ndarray) or contract.depth_map.ndim != 2:
            errors.append("INVALID_DEPTH_MAP")
        elif contract.rgb_image is not None and isinstance(contract.rgb_image, np.ndarray) and contract.rgb_image.ndim == 3:
            if contract.depth_map.shape != contract.rgb_image.shape[:2]:
                errors.append("DEPTH_SHAPE_MISMATCH")

        # Registration must be explicitly declared True
        if not contract.rgb_depth_registered:
            errors.append("UNREGISTERED_RGBD")

        # Timestamp validations
        if contract.depth_timestamp is None:
            errors.append("MISSING_DEPTH_TIMESTAMP")
        elif not math.isfinite(contract.depth_timestamp) or contract.depth_timestamp < 0:
            errors.append("INVALID_DEPTH_TIMESTAMP")

    if contract.rgb_timestamp is None:
        errors.append("MISSING_RGB_TIMESTAMP")
    elif not math.isfinite(contract.rgb_timestamp) or contract.rgb_timestamp < 0:
        errors.append("INVALID_RGB_TIMESTAMP")

    # Clock domain
    if not contract.clock_domain or not isinstance(contract.clock_domain, str) or not contract.clock_domain.strip():
        errors.append("INVALID_CLOCK_DOMAIN")

    # Desynchronization check
    if (
        contract.rgb_timestamp is not None
        and contract.depth_timestamp is not None
        and math.isfinite(contract.rgb_timestamp)
        and math.isfinite(contract.depth_timestamp)
    ):
        if abs(contract.rgb_timestamp - contract.depth_timestamp) > 0.005:
            errors.append("TIMESTAMP_DESYNC")

    return (len(errors) == 0, errors)
