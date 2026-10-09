"""Các giới hạn dưới đây chỉ dành cho simulation/test; đơn vị SI."""
from dataclasses import asdict, dataclass, field
import hashlib
import json

@dataclass(frozen=True)
class VisionConfig:
    hsv_lower: tuple = (35, 80, 80)
    hsv_upper: tuple = (90, 255, 255)
    morphology_size: int = 3
    min_component_area: int = 50
    experimental_sync_extrapolation_ms: float = 50.0
    experimental_rgbd_sync_tolerance_ms: float = 25.0
    experimental_state_history_size: int = 64
    experimental_tracking_timeout_ms: float = 250.0
    experimental_observation_hysteresis_ms: float = 40.0
    experimental_prediction_uncertainty_limit_px: float = 80.0
    experimental_watchdog_stale_ms: float = 500.0
    experimental_depth_min_valid_ratio: float = 0.95
    experimental_depth_resample_points: int = 100
    experimental_roi_margin_px: int = 40
    experimental_roi_prediction_expansion_px: int = 30
    experimental_roi_min_size_px: int = 96
    experimental_min_line_separation_px: float = 20.0
    experimental_max_line_separation_px: float = 180.0
    experimental_blur_laplacian_min: float = 4.0
    experimental_min_observation_confidence: float = 0.45

@dataclass(frozen=True)
class PathConfig:
    points: int = 100
    smoothing_window: int = 5

@dataclass(frozen=True)
class MotionConfig:
    max_speed: float = 0.08  # m/s
    acceleration: float = 0.04  # m/s², tangential look-ahead
    lateral_acceleration: float = 0.02  # m/s²

@dataclass(frozen=True)
class SafetyConfig:
    workspace_min: tuple = (-0.6, -0.5, 0.0)
    workspace_max: tuple = (0.6, 0.5, 1.0)
    max_step: float = 0.04  # m
    max_c_step: float = 0.5  # rad
    max_speed: float = 0.08  # m/s

@dataclass(frozen=True)
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    path: PathConfig = field(default_factory=PathConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output_dir: str = "output"
    mode: str = "compatibility"

    def snapshot(self) -> dict:
        """Canonical snapshot for audit/reproduction; no process-random hash()."""
        return asdict(self)

    @property
    def config_version(self) -> str:
        canonical = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
