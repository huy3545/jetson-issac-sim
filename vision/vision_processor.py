"""Normalization → HSV → morphology → thinning → graph tracing."""
from collections import deque
import cv2
import numpy as np
from config import VisionConfig
from data_source import RGBDFrame
from .contracts import (LineVisionResult, ObservationStatus, OverallState,
                        TrackingState, VisionResult)
from .v11 import resample_arc_length, sample_depth_valid

class VisionProcessor:
    def __init__(self, config: VisionConfig):
        self.config = config

    @staticmethod
    def _thin(mask: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning bằng NumPy; không cần opencv-contrib."""
        image = np.pad((mask > 0).astype(np.uint8), 1)
        while True:
            changed = False
            for phase in (0, 1):
                p = image[1:-1, 1:-1]
                n = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
                     image[2:, 2:], image[2:, 1:-1], image[2:, :-2],
                     image[1:-1, :-2], image[:-2, :-2]]
                count = sum(n)
                transitions = sum((n[i] == 0) & (n[(i + 1) % 8] == 1) for i in range(8))
                if phase == 0:
                    a, b = n[0] * n[2] * n[4], n[2] * n[4] * n[6]
                else:
                    a, b = n[0] * n[2] * n[6], n[0] * n[4] * n[6]
                remove = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & (a == 0) & (b == 0)
                if remove.any():
                    p[remove] = 0
                    changed = True
            if not changed:
                return image[1:-1, 1:-1]

    @staticmethod
    def _trace(skeleton: np.ndarray) -> np.ndarray:
        """Trace đường geodesic giữa hai đầu xa nhau trên graph 8-neighbor."""
        nodes = set(map(tuple, np.argwhere(skeleton)))
        if len(nodes) < 2:
            raise ValueError("Skeleton không đủ điểm")
        def walk(start):
            parents = {start: None}
            queue = deque([start])
            last = start
            while queue:
                last = queue.popleft()
                y, x = last
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nxt = (y + dy, x + dx)
                        if nxt in nodes and nxt not in parents:
                            parents[nxt] = last
                            queue.append(nxt)
            return last, parents
        end, parents = walk(min(nodes))
        if len(parents) != len(nodes):
            raise ValueError("Skeleton bị đứt; cần điều chỉnh segmentation")
        end, parents = walk(end)
        route = []
        while end is not None:
            route.append(end)
            end = parents[end]
        line = np.array(route, dtype=float)[:, ::-1]  # (u, v)
        if tuple(line[0, ::-1]) > tuple(line[-1, ::-1]):
            line = line[::-1]
        return line

    def process(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nhận RGB uint8 hoặc float [0,1], trả hai polyline pixel (u,v)."""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("RGB phải có shape HxWx3 và dữ liệu hữu hạn")
        if rgb.dtype != np.uint8:
            if np.min(rgb) < 0 or np.max(rgb) > 1:
                raise ValueError("RGB float phải nằm trong [0,1]")
            rgb = np.rint(rgb * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper))
        k = self.config.morphology_size
        if k < 1 or k % 2 == 0:
            raise ValueError("morphology_size phải là số lẻ dương")
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.config.min_component_area]
        if len(components) != 2:
            raise ValueError(f"Cần đúng hai đường tham chiếu tách rời, tìm thấy {len(components)}")
        lines = [self._trace(self._thin(labels == i)) for i in components]
        lines.sort(key=lambda line: float(line[:, 0].mean()))
        return lines[0], lines[1]

    def process_frame(self, frame: RGBDFrame) -> VisionResult:
        """V1.1 data contract; does not alter the legacy process(rgb) algorithm.

        This is intentionally stateless at checkpoint 1's extraction boundary; the
        per-line state machine is supplied separately for a runtime adapter.
        """
        if frame.rgb is None or frame.depth is None or frame.rgb.shape[:2] != frame.depth.shape:
            raise ValueError("RGB/Depth frame mismatch")
        if not frame.rgb_depth_registered:
            raise ValueError("Unregistered RGB/Depth mapping is not supported")
        if frame.timestamp is not None and (not np.isfinite(frame.timestamp)):
            raise ValueError("Invalid frame timestamp")
        if frame.rgb_timestamp is not None and frame.depth_timestamp is not None:
            if abs(frame.rgb_timestamp-frame.depth_timestamp)*1000 > self.config.experimental_rgbd_sync_tolerance_ms:
                raise ValueError("RGB/Depth timestamp mismatch")
        left, right = self.process(frame.rgb)
        def make_line(raw):
            sampled=resample_arc_length(raw, self.config.experimental_depth_resample_points)
            valid, ratio, ranges=sample_depth_valid(frame.depth,sampled)
            status=ObservationStatus.OBSERVED if ratio >= self.config.experimental_depth_min_valid_ratio else ObservationStatus.INVALID
            return LineVisionResult(points=sampled, observation_status=status,
                tracking_state=TrackingState.TRACKING if status is ObservationStatus.OBSERVED else TrackingState.LOST,
                confidence=ratio, depth_valid=valid, depth_valid_ratio=ratio, depth_valid_ranges=ranges)
        l,r=make_line(left),make_line(right)
        overall=OverallState.BOTH_OK if l.observation_status is r.observation_status is ObservationStatus.OBSERVED else OverallState.ONE_DEGRADED
        return VisionResult(frame_timestamp=frame.timestamp, frame_clock_domain=frame.clock_domain,
                            left=l,right=r,overall_state=overall)
