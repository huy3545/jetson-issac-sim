"""RGB-D contract độc lập simulator: RGB uint8, depth mét, pixel (u, v)."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import cv2

@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: np.ndarray

    def __post_init__(self):
        values = np.array([self.fx, self.fy, self.cx, self.cy])
        transform = np.asarray(self.camera_to_world, dtype=float)
        if not np.isfinite(values).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsics không hợp lệ")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_world phải là ma trận hữu hạn 4x4")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Transform phải có hàng cuối [0, 0, 0, 1]")
        self.camera_to_world = transform

@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    calibration: CameraCalibration
    # Optional V1.1 metadata. Legacy V1 callers need only the fields above.
    timestamp: float | None = None
    clock_domain: str = "legacy"
    frame_id: str | int | None = None
    rgb_timestamp: float | None = None
    depth_timestamp: float | None = None
    rgb_depth_registered: bool = True
    coordinate_frame: str = "camera"
    occlusion_mask: np.ndarray | None = None
    wheel_state: dict | None = None
    fabric_movement_uncertainty: float | None = None

class RGBDDataSource(Protocol):
    def read(self) -> RGBDFrame:
        """Trả RGB và depth đã đăng ký cùng hệ pixel, depth dọc trục camera Z."""
        ...

class TestDataSource:
    """Sinh hai vạch xanh trên nền tối; không giả lập camera phần cứng."""
    def __init__(self, scenario: str = "curve"):
        self.scenario = scenario

    def read(self) -> RGBDFrame:
        if self.scenario not in ("straight", "curve"):
            raise ValueError(f"Unknown test scenario: {self.scenario}")
        height, width = 360, 480
        rgb = np.full((height, width, 3), 25, dtype=np.uint8)
        v = np.linspace(35, height - 36, 200)
        offset = 35 * np.sin(np.linspace(0, 2 * np.pi, len(v))) if self.scenario == "curve" else np.zeros_like(v)
        for shift in (-40, 40):
            pixels = np.column_stack((width / 2 + shift + offset, v)).astype(np.int32)
            cv2.polylines(rgb, [pixels], False, (0, 220, 0), 7)
        depth = np.full((height, width), 0.5, dtype=float)
        camera = CameraCalibration(450, 450, width / 2, height / 2, np.eye(4))
        return RGBDFrame(rgb, depth, camera)
