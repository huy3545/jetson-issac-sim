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
class PointDerivationMethod(str, Enum):
    DIRECT="DIRECT"; RESAMPLING="RESAMPLING"; PREDICTION="PREDICTION"; UNAVAILABLE="UNAVAILABLE"

VISION_POINT_PROVENANCE_SCHEMA_VERSION = "1.1"

@dataclass(frozen=True)
class VisionPointProvenance:
    """Lineage published for a pixel point, without inventing sensor evidence."""
    line_id: str
    source_index: int | None
    pixel_xy: tuple[float, float] | None
    observation_status: ObservationStatus
    tracking_state: TrackingState
    derivation_method: PointDerivationMethod
    source_references: tuple[str, ...] = ()
    lineage_available: bool = False
    confidence_reference: str | None = None
    timestamp: float | None = None
    clock_domain: str | None = None
    source_segment_start_index: int | None = None
    source_segment_end_index: int | None = None
    interpolation_fraction: float | None = None
    source_segment_start_xy: tuple[float, float] | None = None
    source_segment_end_xy: tuple[float, float] | None = None
    schema_version: str = VISION_POINT_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != VISION_POINT_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported vision point provenance schema")
        if self.line_id not in ("left", "right"):
            raise ValueError("line_id must be left or right")
        if self.source_index is not None and (isinstance(self.source_index, bool) or self.source_index < 0):
            raise ValueError("source_index must be non-negative")
        if self.pixel_xy is not None:
            value = np.asarray(self.pixel_xy, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("pixel_xy must be two finite coordinates")
            object.__setattr__(self, "pixel_xy", (float(value[0]), float(value[1])))
        if not isinstance(self.observation_status, ObservationStatus):
            raise ValueError("unknown observation_status")
        if not isinstance(self.tracking_state, TrackingState):
            raise ValueError("unknown tracking_state")
        if not isinstance(self.derivation_method, PointDerivationMethod):
            raise ValueError("unknown derivation_method")
        if not isinstance(self.source_references, tuple) or not all(isinstance(item, str) and item for item in self.source_references):
            raise ValueError("source_references must be non-empty strings")
        if self.timestamp is not None and not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite when supplied")
        for index in (self.source_segment_start_index, self.source_segment_end_index):
            if index is not None and (isinstance(index, bool) or index < 0):
                raise ValueError("source segment indices must be non-negative")
        if self.interpolation_fraction is not None and (not np.isfinite(self.interpolation_fraction) or not 0. <= self.interpolation_fraction <= 1.):
            raise ValueError("interpolation_fraction must be in [0, 1]")
        for name in ("source_segment_start_xy", "source_segment_end_xy"):
            value=getattr(self,name)
            if value is not None:
                array=np.asarray(value,dtype=float)
                if array.shape!=(2,) or not np.isfinite(array).all(): raise ValueError(f"{name} must be finite xy")
                object.__setattr__(self,name,(float(array[0]),float(array[1])))

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
    point_provenance: tuple[VisionPointProvenance, ...] | None = None
    provenance_diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if self.points is not None:
            array = np.ascontiguousarray(np.array(self.points, dtype=float, copy=True))
            points = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, "points", points)
        if self.depth_valid is not None:
            array = np.ascontiguousarray(np.array(self.depth_valid, dtype=bool, copy=True))
            valid = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, "depth_valid", valid)
        if self.point_provenance is not None:
            if not isinstance(self.point_provenance, tuple) or not all(isinstance(item, VisionPointProvenance) for item in self.point_provenance):
                raise ValueError("point_provenance must contain VisionPointProvenance")
            if self.points is None or len(self.point_provenance) != len(self.points):
                raise ValueError("point_provenance must align with points")
        if not isinstance(self.provenance_diagnostics, tuple) or not all(isinstance(item, str) and item for item in self.provenance_diagnostics):
            raise ValueError("provenance_diagnostics must be non-empty strings")

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
    calibration_status: str = "UNVERIFIED"
    calibration_id: str | None = None
