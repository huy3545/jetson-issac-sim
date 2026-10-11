"""Read-only consumer for qualified synthetic Path geometry."""
from uuid import uuid4
import numpy as np
from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from .six_module_dry_run import DryRunStatus
from .isaac_preview_contracts import IsaacPreviewArtifact, PREVIEW_SCHEMA_VERSION

SQUARE_MITER_FIXTURE = "stage7c6-square-miter-90"
DISPLAY_TRANSFORM = {"kind": "DISPLAY_ONLY_XY_TO_XYZ", "usd_up_axis": "Z", "z_preview_m": 0.0,
                     "scale_m_per_stage_unit": 1.0, "physical_transform_verified": False}


class IsaacPreviewEvidenceAdapter:
    def __init__(self, path_adapter=None):
        self.path_adapter = path_adapter or VisionPathAdapter(solver=CornerAwareTwoLineSolver(), require_point_provenance=True)

    def build(self, vision, dry_run, *, fixture_id=SQUARE_MITER_FIXTURE, preview_id=None):
        reasons=[]
        if fixture_id != SQUARE_MITER_FIXTURE: reasons.append("UNSUPPORTED_SYNTHETIC_FIXTURE")
        if dry_run.run_id == "" or dry_run.correlation_id == "": reasons.append("TRACE_IDENTITY_INVALID")
        vision_record = dry_run.records[0] if len(dry_run.records) == 6 else None
        if vision_record is None or vision_record.run_id != dry_run.run_id or vision_record.correlation_id != dry_run.correlation_id or vision_record.source_frame_id != vision.frame_id or vision_record.timestamp != vision.frame_timestamp or vision_record.clock_domain != vision.frame_clock_domain:
            reasons.append("TRACE_SOURCE_IDENTITY_MISMATCH")
        preflight=self.path_adapter.validate_observation(vision)
        if preflight: reasons.extend(preflight)
        if reasons: return self._rejected(vision, dry_run, fixture_id, tuple(dict.fromkeys(reasons)), preview_id)
        result=self.path_adapter.plan(vision)
        if result.status is not VisionPathStatus.VERIFIED or not result.solver_result or not result.solver_result.fully_covered:
            return self._rejected(vision, dry_run, fixture_id, result.reason_codes or ("PREVIEW_GEOMETRY_REJECTED",), preview_id,
                                  diagnostics=tuple(getattr(result.solver_result.diagnostics, "excluded", ()) if result.solver_result else ()))
        segment=result.solver_result.segments[0] if len(result.solver_result.segments) == 1 else None
        if segment is None or not np.all(segment.point_valid) or not np.isfinite(segment.xy).all():
            return self._rejected(vision, dry_run, fixture_id, ("INCOMPLETE_SOLVER_EVIDENCE",), preview_id)
        return IsaacPreviewArtifact(PREVIEW_SCHEMA_VERSION, preview_id or uuid4().hex, dry_run.run_id, dry_run.correlation_id,
            vision.frame_id, vision.frame_timestamp, vision.frame_clock_domain, vision.coordinate_frame, vision.units,
            fixture_id, self._tuple(vision.left.points), self._tuple(vision.right.points), self._tuple(segment.xy),
            "SYNTHETIC_GEOMETRY_VERIFIED", tuple(result.solver_result.diagnostics.excluded), "POINT_PROVENANCE_AVAILABLE",
            dict(DISPLAY_TRANSFORM), True, False, ("STRICT_GATE_B_BLOCKED",))

    def _rejected(self, vision, dry_run, fixture_id, reasons, preview_id, diagnostics=()):
        return IsaacPreviewArtifact(PREVIEW_SCHEMA_VERSION, preview_id or uuid4().hex, dry_run.run_id, dry_run.correlation_id,
            vision.frame_id, vision.frame_timestamp, vision.frame_clock_domain, vision.coordinate_frame, vision.units,
            fixture_id, self._tuple(vision.left.points), self._tuple(vision.right.points), None, "REJECTED",
            tuple(diagnostics), "UNAVAILABLE", dict(DISPLAY_TRANSFORM), False, False, tuple(reasons))
    @staticmethod
    def _tuple(points): return tuple((float(point[0]),float(point[1])) for point in points)
