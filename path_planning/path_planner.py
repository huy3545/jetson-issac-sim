"""Resample hai đường theo arc length, ghép điểm và chiếu RGB-D sang 3D."""
import numpy as np
from config import PathConfig
from data_source import CameraCalibration

class PathPlanner:
    def __init__(self, config: PathConfig):
        self.config = config

    def _resample(self, line):
        line = np.asarray(line, dtype=float)
        if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 or not np.isfinite(line).all():
            raise ValueError("Reference line phải là Nx2 hữu hạn, ít nhất hai điểm")
        keep = np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-9]
        line = line[keep]
        if len(line) < 2 or self.config.points < 2:
            raise ValueError("Không đủ điểm hoặc chiều dài đường bằng 0")
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        samples = np.linspace(0, distance[-1], self.config.points)
        return np.column_stack([np.interp(samples, distance, line[:, j]) for j in range(2)])

    def plan(self, left_line, right_line, depth, calibration: CameraCalibration):
        left = self._resample(left_line)
        right = self._resample(right_line)
        same = np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1])
        reverse = np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0])
        if reverse < same:
            right = right[::-1]
        center = (left + right) / 2
        window = self.config.smoothing_window
        if window < 1 or window % 2 == 0:
            raise ValueError("smoothing_window phải là số lẻ dương")
        # Moving average nhẹ; giữ nguyên endpoint.
        if window > 1:
            padded = np.pad(center, ((window // 2, window // 2), (0, 0)), mode="edge")
            smooth = np.column_stack([np.convolve(padded[:, j], np.ones(window) / window, mode="valid") for j in range(2)])
            smooth[[0, -1]] = center[[0, -1]]
            center = smooth
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 2:
            raise ValueError("Depth phải là HxW, đơn vị mét")
        u, v = center.T
        if np.any(u < 0) or np.any(u > depth.shape[1] - 1) or np.any(v < 0) or np.any(v > depth.shape[0] - 1):
            raise ValueError("Centerline nằm ngoài depth image")
        # Bilinear sampling giữ độ liên tục; từ chối depth hole thay vì đoán.
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        u1, v1 = np.minimum(u0 + 1, depth.shape[1] - 1), np.minimum(v0 + 1, depth.shape[0] - 1)
        neighbors = np.array([depth[v0, u0], depth[v0, u1], depth[v1, u0], depth[v1, u1]])
        if not np.isfinite(neighbors).all() or (neighbors <= 0).any():
            raise ValueError("Depth tại centerline chứa NaN/Inf hoặc giá trị <= 0")
        a, b = u - u0, v - v0
        z = neighbors[0]*(1-a)*(1-b) + neighbors[1]*a*(1-b) + neighbors[2]*(1-a)*b + neighbors[3]*a*b
        camera_points = np.column_stack(((u-calibration.cx)*z/calibration.fx, (v-calibration.cy)*z/calibration.fy, z, np.ones(len(z))))
        return (camera_points @ calibration.camera_to_world.T)[:, :3]
