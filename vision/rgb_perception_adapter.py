"""RGB perception adapter for physical image ingestion and metric projection.

Responsible for:
1. Phase A (RGB Perception): Ingests raw RGB images, verifies sharpness and
   geometry, extracts pixel polylines via VisionProcessor, and generates
   evidence-grounded confidence scores (never hardcoded values).
2. Phase B (RGB-D Metric Projection): Projects pixel boundaries to metric
   coordinates in 'workpiece_m' strictly when valid calibration and registered
   depth evidence exist. Never invents synthetic transforms.

Strictly non-executable (execution_authorized=False) by design.
"""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence
import cv2
import numpy as np

from config import VisionConfig
from data_source import CameraCalibration, RGBDFrame
from vision.contracts import (
    ContourType,
    LineVisionResult,
    ObservationStatus,
    OrientationSource,
    OverallState,
    PointDerivationMethod,
    TrackingState,
    VisionPointProvenance,
    VisionResult,
)
from vision.v11 import resample_arc_length, sample_depth_valid
from vision.vision_processor import VisionProcessor


class CalibrationEvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INVALID = "INVALID"
    MISSING = "MISSING"


@dataclass(frozen=True)
class PerceptionQuality:
    """Quality metrics computed from raw image perception."""
    laplacian_variance: float
    is_sharp: bool
    component_count: int
    mean_separation_px: float
    quality_score: float
    diagnostics: tuple[str, ...] = ()
    is_confidence_calibrated: bool = False
    calibrated_confidence: float | None = None

    @property
    def calculated_confidence(self) -> float:
        """Backward-compatibility alias for quality_score."""
        return self.quality_score


class RGBPerceptionAdapter:
    """Bridges raw camera frames (RGB/RGB-D) into qualified VisionResult contracts."""

    def __init__(self, config: VisionConfig | None = None, min_blur_var: float = 10.0):
        self.config = config or VisionConfig()
        self.min_blur_var = float(min_blur_var)
        self.processor = VisionProcessor(self.config)

    def assess_quality(self, rgb: np.ndarray) -> tuple[float, bool]:
        """Compute image sharpness via Laplacian variance."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB array must be HxWx3")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = lap_var >= self.min_blur_var
        return lap_var, is_sharp

    def _compute_quality_score(
        self,
        lap_var: float,
        pts_left: np.ndarray,
        pts_right: np.ndarray,
        depth_ratio: float = 1.0,
    ) -> float:
        """Dynamically calculate heuristic quality score based on image sharpness, length, and depth validity.

        This is an empirical image quality heuristic, NOT a statistically calibrated confidence score.
        """
        # Sharpness score in [0.2, 1.0]
        sharpness_score = float(np.clip(lap_var / (self.min_blur_var * 4.0), 0.2, 1.0))
        # Point count completeness score in [0.5, 1.0]
        min_pts = min(len(pts_left), len(pts_right)) if len(pts_left) and len(pts_right) else 0
        length_score = float(np.clip(min_pts / 100.0, 0.5, 1.0))
        # Depth valid ratio in [0.0, 1.0]
        depth_score = float(np.clip(depth_ratio, 0.0, 1.0))

        quality = sharpness_score * 0.4 + length_score * 0.2 + depth_score * 0.4
        return float(np.clip(quality, 0.05, 0.98))

    def _compute_confidence(
        self,
        lap_var: float,
        pts_left: np.ndarray,
        pts_right: np.ndarray,
        depth_ratio: float = 1.0,
    ) -> float:
        """Backward-compatibility alias for _compute_quality_score."""
        return self._compute_quality_score(lap_var, pts_left, pts_right, depth_ratio=depth_ratio)

    def process_rgb(
        self,
        rgb: np.ndarray,
        *,
        timestamp_s: float = 0.0,
        frame_id: str = "rgb-perception-frame",
        clock_domain: str = "camera_clock",
    ) -> tuple[VisionResult, PerceptionQuality]:
        """Phase A: Extract 2D pixel lines from raw RGB image.

        Units are explicitly 'pixels' and coordinate_frame is 'camera'.
        """
        lap_var, is_sharp = self.assess_quality(rgb)
        diagnostics = []
        if not is_sharp:
            diagnostics.append("IMAGE_BLURRY")

        try:
            left_px, right_px = self.processor.process(rgb)
        except ValueError as exc:
            # Handle 0 or 1 line or segmentation failure gracefully
            err_msg = str(exc)
            diag_reason = (
                "INSUFFICIENT_LINES"
                if "đường tham chiếu" in err_msg or "Skeleton" in err_msg
                else "SEGMENTATION_FAILED"
            )
            quality = PerceptionQuality(
                laplacian_variance=lap_var,
                is_sharp=is_sharp,
                component_count=0,
                mean_separation_px=0.0,
                quality_score=0.0,
                diagnostics=tuple([diag_reason] + diagnostics),
                is_confidence_calibrated=False,
                calibrated_confidence=None,
            )
            empty_line = LineVisionResult(
                points=np.empty((0, 2)),
                observation_status=ObservationStatus.INVALID,
                tracking_state=TrackingState.LOST,
                confidence=0.0,
                verified=False,
            )
            res = VisionResult(
                frame_timestamp=timestamp_s,
                frame_clock_domain=clock_domain,
                frame_id=frame_id,
                coordinate_frame="camera",
                units="pixels",
                left=empty_line,
                right=empty_line,
                overall_state=OverallState.BOTH_LOST,
            )
            return res, quality

        # Compute line separation
        min_len = min(len(left_px), len(right_px))
        sep = float(np.mean(np.linalg.norm(left_px[:min_len] - right_px[:min_len], axis=1)))
        quality_score = self._compute_quality_score(lap_var, left_px, right_px, depth_ratio=1.0)

        quality = PerceptionQuality(
            laplacian_variance=lap_var,
            is_sharp=is_sharp,
            component_count=2,
            mean_separation_px=sep,
            quality_score=quality_score,
            diagnostics=tuple(diagnostics),
            is_confidence_calibrated=False,
            calibrated_confidence=None,
        )

        def _make_line(name: str, pts: np.ndarray) -> LineVisionResult:
            prov = tuple(
                VisionPointProvenance(
                    line_id=name,
                    source_index=i,
                    pixel_xy=(float(p[0]), float(p[1])),
                    observation_status=ObservationStatus.OBSERVED,
                    tracking_state=TrackingState.TRACKING,
                    derivation_method=PointDerivationMethod.DIRECT,
                    source_references=(f"{name}:{i}",),
                    lineage_available=True,
                    timestamp=timestamp_s,
                    clock_domain=clock_domain,
                )
                for i, p in enumerate(pts)
            )
            return LineVisionResult(
                points=pts,
                observation_status=ObservationStatus.OBSERVED,
                tracking_state=TrackingState.TRACKING,
                confidence=quality_score,
                depth_valid=np.ones(len(pts), bool),
                orientation_source=OrientationSource.REFERENCE,
                verified=True,
                point_provenance=prov,
            )

        res = VisionResult(
            frame_timestamp=timestamp_s,
            frame_clock_domain=clock_domain,
            frame_id=frame_id,
            coordinate_frame="camera",
            units="pixels",
            left=_make_line("left", left_px),
            right=_make_line("right", right_px),
            overall_state=OverallState.BOTH_OK,
        )
        return res, quality

    def process_rgbd(
        self,
        rgb: np.ndarray | RGBDFrame,
        depth: np.ndarray | None = None,
        calibration: CameraCalibration | None = None,
        *,
        timestamp_s: float = 0.0,
        frame_id: str = "rgbd-perception-frame",
        clock_domain: str = "camera_clock",
        rgb_timestamp: float | None = None,
        depth_timestamp: float | None = None,
        rgb_depth_registered: bool = True,
        max_sync_delta_s: float = 0.005,
    ) -> tuple[VisionResult, PerceptionQuality]:
        """Phase B: Project pixel boundaries to metric workpiece coordinates.

        Strict V1.2 Data Integrity:
        - Rejects unregistered frames (UNREGISTERED_RGBD).
        - Rejects desynchronized RGB-Depth timestamps (TIMESTAMP_DESYNC).
        - Rejects invalid or empty clock domains (INVALID_CLOCK_DOMAIN).
        - Never replaces invalid depth with fallback constants (e.g. 0.5m).
        - Propagates calibration status and id through provenance and VisionResult.
        """
        # Support ingestion from RGBDFrame directly
        if isinstance(rgb, RGBDFrame):
            frame = rgb
            rgb = frame.rgb
            depth = frame.depth
            calibration = frame.calibration
            timestamp_s = frame.timestamp if frame.timestamp is not None else timestamp_s
            clock_domain = frame.clock_domain
            frame_id = str(frame.frame_id) if frame.frame_id is not None else frame_id
            rgb_timestamp = frame.rgb_timestamp
            depth_timestamp = frame.depth_timestamp
            rgb_depth_registered = frame.rgb_depth_registered

        # 1. Validation: Registration check
        if not rgb_depth_registered:
            raise ValueError("UNREGISTERED_RGBD: RGB and Depth frames are not registered")

        # 2. Validation: Clock domain validation
        if not clock_domain or not isinstance(clock_domain, str) or not clock_domain.strip():
            raise ValueError("INVALID_CLOCK_DOMAIN: Clock domain must be specified and non-empty")

        # 3. Validation: Timestamp synchronization check
        if rgb_timestamp is not None and depth_timestamp is not None:
            if not (math.isfinite(rgb_timestamp) and math.isfinite(depth_timestamp)):
                raise ValueError("INVALID_TIMESTAMP: rgb_timestamp and depth_timestamp must be finite numbers")
            sync_delta = abs(rgb_timestamp - depth_timestamp)
            if sync_delta > max_sync_delta_s:
                raise ValueError(
                    f"TIMESTAMP_DESYNC: Delta {sync_delta:.6f}s exceeds max_sync_delta_s ({max_sync_delta_s:.6f}s)"
                )

        # 4. Validation: Depth array and shape
        if depth is None or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"Depth shape {getattr(depth, 'shape', None)} must match RGB shape {rgb.shape[:2]}"
            )

        # 5. Validation: Calibration object
        if not isinstance(calibration, CameraCalibration):
            raise ValueError("Invalid CameraCalibration object")
        if min(calibration.fx, calibration.fy) <= 0 or not np.isfinite(calibration.camera_to_world).all():
            raise ValueError("CameraCalibration values must be positive and finite")

        # Step 1: Extract pixel lines
        pixel_result, quality = self.process_rgb(
            rgb, timestamp_s=timestamp_s, frame_id=frame_id, clock_domain=clock_domain
        )
        if pixel_result.left.observation_status is not ObservationStatus.OBSERVED:
            return pixel_result, quality

        left_px = pixel_result.left.points
        right_px = pixel_result.right.points

        # Step 2: Project each line to metric 3D using pinhole projection without fallback
        def _project_line(name: str, pts_px: np.ndarray) -> tuple[LineVisionResult, list[str]]:
            line_diagnostics: list[str] = []
            sampled_px = resample_arc_length(pts_px, self.config.experimental_depth_resample_points)
            valid_mask, _, valid_ranges = sample_depth_valid(depth, sampled_px)

            u = sampled_px[:, 0]
            v = sampled_px[:, 1]
            u_int = np.clip(np.round(u).astype(int), 0, depth.shape[1] - 1)
            v_int = np.clip(np.round(v).astype(int), 0, depth.shape[0] - 1)
            z_vals = depth[v_int, u_int]

            # Invalidate any non-positive or non-finite depth
            z_valid = valid_mask & np.isfinite(z_vals) & (z_vals > 0.0)
            actual_valid_count = int(np.sum(z_valid))
            actual_valid_ratio = actual_valid_count / len(z_valid) if len(z_valid) > 0 else 0.0

            has_depth_gap = len(valid_ranges) > 1
            has_invalid_depth = (actual_valid_ratio < 1.0) or not np.all(z_valid)

            if has_depth_gap:
                line_diagnostics.append("UNRESOLVED_DEPTH_GAP")
            if actual_valid_ratio < self.config.experimental_depth_min_valid_ratio or (
                has_invalid_depth and not has_depth_gap
            ):
                line_diagnostics.append("INVALID_DEPTH_DATA")

            # Check if metric projection must be rejected:
            # - Ratio below safety threshold
            # - Or broken boundary line with gap (no bridging allowed)
            # - Or zero valid depth points
            if (
                actual_valid_ratio < self.config.experimental_depth_min_valid_ratio
                or has_depth_gap
                or actual_valid_count == 0
            ):
                # REJECT metric projection: zero 3D points emitted to path planning
                return LineVisionResult(
                    points=np.empty((0, 2)),
                    observation_status=ObservationStatus.INVALID,
                    tracking_state=TrackingState.LOST,
                    confidence=0.0,
                    depth_valid=z_valid,
                    depth_valid_ratio=actual_valid_ratio,
                    depth_valid_ranges=valid_ranges,
                    orientation_source=OrientationSource.REFERENCE,
                    verified=False,
                    point_provenance=(),
                    provenance_diagnostics=tuple(line_diagnostics),
                ), line_diagnostics

            # All points or single contiguous valid segment:
            # Strictly select ONLY valid depth points (NO 0.5 fallback)
            valid_indices = np.where(z_valid)[0]
            u_valid = u[valid_indices]
            v_valid = v[valid_indices]
            z_valid_vals = z_vals[valid_indices]
            px_valid = sampled_px[valid_indices]

            # Pinhole projection to camera coordinates
            x_cam = (u_valid - calibration.cx) * z_valid_vals / calibration.fx
            y_cam = (v_valid - calibration.cy) * z_valid_vals / calibration.fy
            cam_pts_homo = np.column_stack((x_cam, y_cam, z_valid_vals, np.ones(len(z_valid_vals))))

            # Transform to workpiece/world frame: P_world = P_cam @ T^T
            world_pts = (cam_pts_homo @ calibration.camera_to_world.T)[:, :2]

            metric_quality = self._compute_quality_score(
                quality.laplacian_variance, pts_px, pts_px, depth_ratio=actual_valid_ratio
            )

            prov = tuple(
                VisionPointProvenance(
                    line_id=name,
                    source_index=int(valid_indices[i]),
                    pixel_xy=(float(px_valid[i, 0]), float(px_valid[i, 1])),
                    observation_status=ObservationStatus.OBSERVED,
                    tracking_state=TrackingState.TRACKING,
                    derivation_method=PointDerivationMethod.DIRECT,
                    source_references=(
                        f"{name}:{int(valid_indices[i])}",
                        f"calibration_status:{calibration.calibration_status}",
                        f"calibration_id:{calibration.calibration_id}",
                    ),
                    lineage_available=True,
                    timestamp=timestamp_s,
                    clock_domain=clock_domain,
                )
                for i in range(len(world_pts))
            )

            return LineVisionResult(
                points=world_pts,
                observation_status=ObservationStatus.OBSERVED,
                tracking_state=TrackingState.TRACKING,
                confidence=metric_quality,
                depth_valid=z_valid,
                depth_valid_ratio=actual_valid_ratio,
                depth_valid_ranges=valid_ranges,
                orientation_source=OrientationSource.REFERENCE,
                verified=True,
                point_provenance=prov,
                provenance_diagnostics=tuple(line_diagnostics),
            ), line_diagnostics

        metric_left, left_diags = _project_line("left", left_px)
        metric_right, right_diags = _project_line("right", right_px)

        all_diags = list(quality.diagnostics)
        for d in left_diags + right_diags:
            if d not in all_diags:
                all_diags.append(d)

        left_ok = metric_left.observation_status is ObservationStatus.OBSERVED
        right_ok = metric_right.observation_status is ObservationStatus.OBSERVED

        if left_ok and right_ok:
            overall = OverallState.BOTH_OK
        elif left_ok or right_ok:
            overall = OverallState.ONE_LOST
        else:
            overall = OverallState.BOTH_LOST

        combined_depth_ratio = (metric_left.depth_valid_ratio + metric_right.depth_valid_ratio) / 2.0
        final_quality_score = (
            self._compute_quality_score(
                quality.laplacian_variance, left_px, right_px, depth_ratio=combined_depth_ratio
            )
            if (left_ok and right_ok)
            else 0.0
        )

        final_quality = PerceptionQuality(
            laplacian_variance=quality.laplacian_variance,
            is_sharp=quality.is_sharp,
            component_count=quality.component_count,
            mean_separation_px=quality.mean_separation_px,
            quality_score=final_quality_score,
            diagnostics=tuple(all_diags),
            is_confidence_calibrated=False,
            calibrated_confidence=None,
        )

        metric_result = VisionResult(
            frame_timestamp=timestamp_s,
            frame_clock_domain=clock_domain,
            frame_id=frame_id,
            coordinate_frame="workpiece_m",
            units="m",
            left=metric_left,
            right=metric_right,
            overall_state=overall,
            calibration_status=calibration.calibration_status,
            calibration_id=calibration.calibration_id,
        )
        return metric_result, final_quality

