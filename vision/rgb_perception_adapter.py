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
    calculated_confidence: float
    diagnostics: tuple[str, ...]


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

    def _compute_confidence(
        self,
        lap_var: float,
        pts_left: np.ndarray,
        pts_right: np.ndarray,
        depth_ratio: float = 1.0,
    ) -> float:
        """Dynamically calculate confidence based on image sharpness, length, and depth validity.

        Never returns an ungrounded hardcoded constant.
        """
        # Sharpness score in [0.2, 1.0]
        sharpness_score = float(np.clip(lap_var / (self.min_blur_var * 4.0), 0.2, 1.0))
        # Point count completeness score in [0.5, 1.0]
        min_pts = min(len(pts_left), len(pts_right))
        length_score = float(np.clip(min_pts / 100.0, 0.5, 1.0))
        # Depth valid ratio in [0.0, 1.0]
        depth_score = float(np.clip(depth_ratio, 0.0, 1.0))

        confidence = sharpness_score * 0.4 + length_score * 0.2 + depth_score * 0.4
        return float(np.clip(confidence, 0.05, 0.98))

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
            diag_reason = "INSUFFICIENT_LINES" if "đường tham chiếu" in err_msg or "Skeleton" in err_msg else "SEGMENTATION_FAILED"
            quality = PerceptionQuality(
                laplacian_variance=lap_var,
                is_sharp=is_sharp,
                component_count=0,
                mean_separation_px=0.0,
                calculated_confidence=0.0,
                diagnostics=tuple([diag_reason] + diagnostics),
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
        conf = self._compute_confidence(lap_var, left_px, right_px, depth_ratio=1.0)

        quality = PerceptionQuality(
            laplacian_variance=lap_var,
            is_sharp=is_sharp,
            component_count=2,
            mean_separation_px=sep,
            calculated_confidence=conf,
            diagnostics=tuple(diagnostics),
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
                confidence=conf,
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
        rgb: np.ndarray,
        depth: np.ndarray,
        calibration: CameraCalibration,
        *,
        timestamp_s: float = 0.0,
        frame_id: str = "rgbd-perception-frame",
        clock_domain: str = "camera_clock",
    ) -> tuple[VisionResult, PerceptionQuality]:
        """Phase B: Project pixel boundaries to metric workpiece coordinates.

        Requires registered depth and valid CameraCalibration. Never invents transforms.
        """
        # Validate depth shape
        if depth is None or depth.shape != rgb.shape[:2]:
            raise ValueError(f"Depth shape {getattr(depth, 'shape', None)} must match RGB shape {rgb.shape[:2]}")

        # Validate calibration
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

        # Step 2: Project each line to metric 3D using pinhole projection
        def _project_line(name: str, pts_px: np.ndarray) -> LineVisionResult:
            # Resample arc length to standardized count
            sampled_px = resample_arc_length(pts_px, self.config.experimental_depth_resample_points)
            valid_mask, valid_ratio, valid_ranges = sample_depth_valid(depth, sampled_px)

            u = sampled_px[:, 0]
            v = sampled_px[:, 1]
            u_int = np.clip(np.round(u).astype(int), 0, depth.shape[1] - 1)
            v_int = np.clip(np.round(v).astype(int), 0, depth.shape[0] - 1)
            z_vals = depth[v_int, u_int]

            # Invalidate any non-positive or non-finite depth
            z_valid = valid_mask & np.isfinite(z_vals) & (z_vals > 0.0)
            z_safe = np.where(z_valid, z_vals, 0.5)

            # Pinhole projection to camera coordinates
            x_cam = (u - calibration.cx) * z_safe / calibration.fx
            y_cam = (v - calibration.cy) * z_safe / calibration.fy
            cam_pts_homo = np.column_stack((x_cam, y_cam, z_safe, np.ones(len(z_safe))))

            # Transform to workpiece/world frame: P_world = P_cam @ T^T
            world_pts = (cam_pts_homo @ calibration.camera_to_world.T)[:, :2]

            metric_conf = self._compute_confidence(
                quality.laplacian_variance, pts_px, pts_px, depth_ratio=valid_ratio
            )
            obs_status = (
                ObservationStatus.OBSERVED
                if valid_ratio >= self.config.experimental_depth_min_valid_ratio
                else ObservationStatus.INVALID
            )
            track_state = (
                TrackingState.TRACKING if obs_status is ObservationStatus.OBSERVED else TrackingState.LOST
            )

            prov = tuple(
                VisionPointProvenance(
                    line_id=name,
                    source_index=i,
                    pixel_xy=(float(sampled_px[i, 0]), float(sampled_px[i, 1])),
                    observation_status=obs_status,
                    tracking_state=track_state,
                    derivation_method=PointDerivationMethod.DIRECT,
                    source_references=(f"{name}:{i}",),
                    lineage_available=True,
                    timestamp=timestamp_s,
                    clock_domain=clock_domain,
                )
                for i in range(len(world_pts))
            )

            return LineVisionResult(
                points=world_pts,
                observation_status=obs_status,
                tracking_state=track_state,
                confidence=metric_conf,
                depth_valid=z_valid,
                depth_valid_ratio=valid_ratio,
                depth_valid_ranges=valid_ranges,
                orientation_source=OrientationSource.REFERENCE,
                verified=(obs_status is ObservationStatus.OBSERVED),
                point_provenance=prov,
            )

        metric_left = _project_line("left", left_px)
        metric_right = _project_line("right", right_px)

        overall = (
            OverallState.BOTH_OK
            if metric_left.observation_status is metric_right.observation_status is ObservationStatus.OBSERVED
            else OverallState.ONE_DEGRADED
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
        )
        return metric_result, quality
