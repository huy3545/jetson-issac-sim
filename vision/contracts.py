"""Stateful V1.1 contracts. Predicted points are deliberately not OBSERVED."""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from machine_state import MachineState, SyncMode

class ObservationStatus(str, Enum): OBSERVED="OBSERVED"; OCCLUDED="OCCLUDED"; INVALID="INVALID"
class TrackingState(str, Enum): TRACKING="TRACKING"; PREDICTING="PREDICTING"; LOST="LOST"
class OrientationSource(str, Enum): GEOMETRIC="GEOMETRIC"; C_AXIS="C_AXIS"; REFERENCE="REFERENCE"; AMBIGUOUS="AMBIGUOUS"
class OverallState(str, Enum): BOTH_OK="BOTH_OK"; ONE_DEGRADED="ONE_DEGRADED"; ONE_LOST="ONE_LOST"; BOTH_LOST="BOTH_LOST"
class ContourType(str, Enum): OPEN="OPEN"; CLOSED="CLOSED"; UNKNOWN="UNKNOWN"

@dataclass(frozen=True)
class LineVisionResult:
    points: np.ndarray | None = None
    observation_status: ObservationStatus = ObservationStatus.OBSERVED
    tracking_state: TrackingState = TrackingState.TRACKING
    confidence: float = 1.0
    tracking_age_ms: float = 0.0
    depth_valid: np.ndarray | None = None
    depth_valid_ratio: float = 1.0
    depth_valid_ranges: tuple[tuple[int,int], ...] = ()
    orientation_source: OrientationSource = OrientationSource.REFERENCE
    prediction_uncertainty_px: float = 0.0
    verified: bool = True
    missing_ranges: tuple[tuple[int,int], ...] = ()
    contour_type: ContourType = ContourType.OPEN

@dataclass(frozen=True)
class VisionResult:
    frame_timestamp: float | None = None
    frame_clock_domain: str = "legacy"
    machine_state_at_frame: MachineState | None = None
    sync_mode: SyncMode = SyncMode.INTERPOLATED
    sync_age_ms: float | None = 0.0
    time_sync_valid: bool = True
    left: LineVisionResult = field(default_factory=LineVisionResult)
    right: LineVisionResult = field(default_factory=LineVisionResult)
    overall_state: OverallState = OverallState.BOTH_OK
    frame_id: str | int | None = None
    coordinate_frame: str = "camera"
    units: str = "pixels"
    camera_to_reference: np.ndarray | None = None
    occlusion_status: str = "UNKNOWN"
    fabric_movement_uncertainty: float | None = None
