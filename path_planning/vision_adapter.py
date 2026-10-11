"""Experimental, conservative bridge from Vision V1.1 to planar Phase 1a."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from vision.contracts import ContourType, ObservationStatus, OrientationSource, TrackingState, VisionResult
from .planar_medial_axis import PlanarMedialAxisSolver, SolverResult, SolverStatus

class VisionPathStatus(str, Enum): VERIFIED="VERIFIED"; REJECTED="REJECTED"; PARTIAL="PARTIAL"
@dataclass(frozen=True)
class VisionPathResult:
    status: VisionPathStatus
    solver_result: SolverResult|None
    reason_codes: tuple[str,...]
    input_frame_id: str|int|None
    input_timestamp: float|None
    coordinate_frame: str

class VisionPathAdapter:
    """Accept only observed, verified metric boundaries; never invent a transform."""
    def __init__(self, solver: PlanarMedialAxisSolver|None=None, min_confidence=.45, require_point_provenance=False):
        self.solver=solver or PlanarMedialAxisSolver(); self.min_confidence=min_confidence; self.require_point_provenance=require_point_provenance
    def _validate(self,line,name):
        reasons=[]
        if line.points is None: reasons.append(f"{name}_MISSING")
        if line.observation_status is not ObservationStatus.OBSERVED: reasons.append(f"{name}_NOT_OBSERVED")
        if line.tracking_state is not TrackingState.TRACKING: reasons.append(f"{name}_NOT_TRACKING")
        if not line.verified: reasons.append(f"{name}_NOT_VERIFIED")
        if line.confidence<self.min_confidence: reasons.append(f"{name}_LOW_CONFIDENCE")
        if line.prediction_uncertainty_px>0: reasons.append(f"{name}_PREDICTED")
        if line.missing_ranges: reasons.append(f"{name}_GAPPED")
        if line.depth_valid is not None and not np.all(line.depth_valid): reasons.append(f"{name}_DEPTH_INVALID")
        if line.orientation_source is OrientationSource.AMBIGUOUS: reasons.append("AMBIGUOUS_IDENTITY")
        if line.contour_type is ContourType.CLOSED: reasons.append("CLOSED_CONTOUR_UNSUPPORTED")
        if "RESAMPLING_CORNER_PRESERVATION_LIMITED" in line.provenance_diagnostics:
            reasons.append(f"{name}_CORNER_PRESERVATION_LIMITED")
        if self.require_point_provenance:
            if line.point_provenance is None or not all(point.lineage_available for point in line.point_provenance):
                reasons.append(f"{name}_POINT_PROVENANCE_UNAVAILABLE")
        return reasons
    def plan(self, vision: VisionResult)->VisionPathResult:
        reasons=[]
        if vision.frame_timestamp is None: reasons.append("MISSING_TIMESTAMP")
        if not vision.time_sync_valid: reasons.append("TIME_SYNC_INVALID")
        if vision.units!="m": reasons.append("UNITS_NOT_METRIC")
        if vision.coordinate_frame in ("", "camera", "legacy"): reasons.append("COORDINATE_FRAME_UNVERIFIED")
        reasons += self._validate(vision.left,"LEFT") + self._validate(vision.right,"RIGHT")
        if not reasons:
            # Independent conservative boundary-contact gate before the Phase 1a solver.
            for point in np.vstack((vision.left.points,vision.right.points)):
                other=vision.right.points if any(np.array_equal(point,p) for p in vision.left.points) else vision.left.points
                for a,b in zip(other[:-1],other[1:]):
                    vector=b-a; t=np.clip(np.dot(point-a,vector)/np.dot(vector,vector),0.,1.)
                    if np.linalg.norm(point-(a+t*vector)) <= self.solver.config.experimental_touch_tol_m:
                        reasons.append("BOUNDARY_TOUCH"); break
                if reasons: break
        if reasons: return VisionPathResult(VisionPathStatus.REJECTED,None,tuple(sorted(set(reasons))),vision.frame_id,vision.frame_timestamp,vision.coordinate_frame)
        result=self.solver.solve(vision.left.points,vision.right.points)
        status=VisionPathStatus.VERIFIED if result.overall_status is SolverStatus.CONVERGED else VisionPathStatus.REJECTED
        return VisionPathResult(status,result,() if status is VisionPathStatus.VERIFIED else tuple(result.diagnostics.excluded),vision.frame_id,vision.frame_timestamp,vision.coordinate_frame)
