"""Tangent, C-axis radian liên tục, curvature và look-ahead tốc độ."""
import numpy as np
from config import MotionConfig

class MotionPlanner:
    def __init__(self, config: MotionConfig):
        self.config = config

    def plan(self, path):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError("Path phải là Nx3 hữu hạn với ít nhất hai điểm")
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        if (ds <= 1e-9).any():
            raise ValueError("Path có waypoint trùng nhau")
        cfg = self.config
        if min(cfg.max_speed, cfg.acceleration, cfg.lateral_acceleration) <= 0:
            raise ValueError("Motion limits phải dương")
        s = np.r_[0, np.cumsum(ds)]
        tangent = np.gradient(path, s, axis=0)
        if (np.linalg.norm(tangent[:, :2], axis=1) <= 1e-9).any():
            raise ValueError("Không xác định được C-axis cho tangent XY bằng 0")
        c = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        unit = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        curvature = np.linalg.norm(np.gradient(unit, s, axis=0), axis=1)
        speed = np.minimum(cfg.max_speed, np.sqrt(cfg.lateral_acceleration / np.maximum(curvature, 1e-12)))
        speed[0] = speed[-1] = 0.0
        # Backward pass phanh trước cua/điểm cuối; forward pass giới hạn tăng tốc.
        for i in range(len(speed)-2, -1, -1):
            speed[i] = min(speed[i], np.sqrt(speed[i+1]**2 + 2*cfg.acceleration*ds[i]))
        for i in range(1, len(speed)):
            speed[i] = min(speed[i], np.sqrt(speed[i-1]**2 + 2*cfg.acceleration*ds[i-1]))
        return [dict(x=float(p[0]), y=float(p[1]), z=float(p[2]), c=float(angle), speed=float(velocity))
                for p, angle, velocity in zip(path, c, speed)]
