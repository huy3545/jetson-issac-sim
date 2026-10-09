"""Validation chỉ áp dụng các giới hạn mô phỏng cấu hình."""
from dataclasses import dataclass
import numpy as np
from config import SafetyConfig

@dataclass
class ValidationResult:
    valid: bool
    reason: str

    @property
    def status(self):
        return "VALID" if self.valid else "INVALID"

class SafetyMonitor:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def validate(self, targets) -> ValidationResult:
        if targets is None or len(targets) == 0:
            return ValidationResult(False, "Path rỗng")
        try:
            values = np.array([[t[k] for k in ("x", "y", "z", "c", "speed")] for t in targets], dtype=float)
        except (KeyError, TypeError, ValueError):
            return ValidationResult(False, "Waypoint thiếu field hoặc chứa dữ liệu không phải số")
        if not np.isfinite(values).all():
            return ValidationResult(False, "Waypoint chứa NaN/Inf")
        cfg = self.config
        xyz = values[:, :3]
        if ((xyz < cfg.workspace_min) | (xyz > cfg.workspace_max)).any():
            return ValidationResult(False, "XYZ nằm ngoài workspace mô phỏng")
        if (np.linalg.norm(np.diff(xyz, axis=0), axis=1) > cfg.max_step).any():
            return ValidationResult(False, "Khoảng cách waypoint vượt giới hạn mô phỏng")
        if (np.abs(np.diff(values[:, 3])) > cfg.max_c_step).any():
            return ValidationResult(False, "C-axis thay đổi bất thường")
        if ((values[:, 4] < 0) | (values[:, 4] > cfg.max_speed + 1e-12)).any():
            return ValidationResult(False, "Speed vượt giới hạn mô phỏng")
        return ValidationResult(True, "Các target hợp lệ theo cấu hình mô phỏng")
