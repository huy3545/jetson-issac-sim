"""Read-only preview creation and independent revalidation of transport data."""
from uuid import uuid4
import numpy as np
from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from .six_module_dry_run import DryRunStatus, SCHEMA_VERSION
from .isaac_preview_contracts import IsaacPreviewArtifact, PREVIEW_SCHEMA_VERSION, PreviewValidationResult

SQUARE_MITER_FIXTURE="stage7c6-square-miter-90"
CANONICAL_LEFT=np.array([[0.,.1],[.5,.1],[.5,.6]])
CANONICAL_RIGHT=np.array([[0.,-.1],[.7,-.1],[.7,.6]])
DISPLAY_TRANSFORM={"kind":"DISPLAY_ONLY_XY_TO_XYZ","usd_up_axis":"Z","z_preview_m":0.,"scale_m_per_stage_unit":1.,"physical_transform_verified":False}

class IsaacPreviewEvidenceAdapter:
    def __init__(self,path_adapter=None): self.path_adapter=path_adapter or VisionPathAdapter(solver=CornerAwareTwoLineSolver(),require_point_provenance=True)
    def build(self,vision,dry_run,*,fixture_id=SQUARE_MITER_FIXTURE,preview_id=None):
        reasons=self._trace_reasons(vision,dry_run)
        if fixture_id!=SQUARE_MITER_FIXTURE: reasons.append("UNSUPPORTED_SYNTHETIC_FIXTURE")
        preflight=self.path_adapter.validate_observation(vision); reasons.extend(preflight)
        if reasons: return self._artifact(vision,dry_run,fixture_id,None,tuple(dict.fromkeys(reasons)),preview_id)
        result=self.path_adapter.plan(vision)
        return self._artifact(vision,dry_run,fixture_id,result,result.reason_codes,preview_id)
    def validate(self,artifact,vision,dry_run):
        """Validate untrusted transport against actual source and canonical fixture."""
        reasons=self._trace_reasons(vision,dry_run)
        if artifact.schema_version!=PREVIEW_SCHEMA_VERSION: reasons.append("ARTIFACT_SCHEMA_INVALID")
        if artifact.execution_authorized: reasons.append("EXECUTION_AUTHORIZATION_FORBIDDEN")
        expected={"run_id":dry_run.run_id,"correlation_id":dry_run.correlation_id,"source_frame_id":vision.frame_id,"timestamp":vision.frame_timestamp,"clock_domain":vision.frame_clock_domain,"coordinate_frame":vision.coordinate_frame,"units":vision.units}
        if any(getattr(artifact,field)!=value for field,value in expected.items()): reasons.append("ARTIFACT_SOURCE_IDENTITY_MISMATCH")
        if artifact.synthetic_fixture_id!=SQUARE_MITER_FIXTURE or not np.array_equal(np.asarray(artifact.left_boundary_xy),CANONICAL_LEFT) or not np.array_equal(np.asarray(artifact.right_boundary_xy),CANONICAL_RIGHT): reasons.append("CANONICAL_FIXTURE_MISMATCH")
        if not np.array_equal(np.asarray(artifact.left_boundary_xy),vision.left.points) or not np.array_equal(np.asarray(artifact.right_boundary_xy),vision.right.points): reasons.append("ARTIFACT_VISION_BOUNDARY_MISMATCH")
        if dict(artifact.display_transform)!=DISPLAY_TRANSFORM: reasons.append("DISPLAY_TRANSFORM_MISMATCH")
        preflight=self.path_adapter.validate_observation(vision); reasons.extend(preflight)
        result=self.path_adapter.plan(vision) if not preflight else None
        if result is None or result.status is not VisionPathStatus.VERIFIED or not result.solver_result or not result.solver_result.fully_covered: reasons.append("SOLVER_EVIDENCE_INVALID")
        else:
            segment=result.solver_result.segments[0] if len(result.solver_result.segments)==1 else None
            candidate=np.asarray(artifact.centerline_xy) if artifact.centerline_xy is not None else np.empty((0,2))
            if segment is None or not np.all(segment.point_valid) or candidate.shape!=segment.xy.shape or not np.allclose(candidate,segment.xy,atol=1e-12,rtol=0.): reasons.append("CENTERLINE_SOLVER_MISMATCH")
        expected_diagnostics=tuple(result.solver_result.diagnostics.excluded) if result and result.solver_result else ()
        if artifact.solver_diagnostics!=expected_diagnostics: reasons.append("SOLVER_DIAGNOSTICS_MISMATCH")
        if artifact.geometry_status!="SYNTHETIC_GEOMETRY_VERIFIED" or not artifact.preview_authorized or artifact.source_provenance_status!="POINT_PROVENANCE_AVAILABLE": reasons.append("ARTIFACT_ADMISSION_DECLARATION_MISMATCH")
        if artifact.reason_codes!=("STRICT_GATE_B_BLOCKED",): reasons.append("ARTIFACT_REASON_CODES_MISMATCH")
        path=dry_run.records[1]
        expected_path=("STRICT_GATE_B_BLOCKED","CALIBRATION_UNVERIFIED","TRANSFORM_UNVERIFIED","XYZ_UNCERTAINTY_UNKNOWN")
        if path.status is not DryRunStatus.BLOCKED or path.geometry_status!="SYNTHETIC_GEOMETRY_VERIFIED" or not all(reason in path.reason_codes for reason in expected_path): reasons.append("TRACE_PATH_STATUS_INVALID")
        return PreviewValidationResult(artifact,not reasons,tuple(dict.fromkeys(reasons)))
    def _trace_reasons(self,vision,trace):
        reasons=[]
        if getattr(trace,"schema_version",None)!=SCHEMA_VERSION or len(getattr(trace,"records",()))!=6: return ["TRACE_SCHEMA_INVALID"]
        if tuple(record.stage_id for record in trace.records)!=("vision","path","motion","safety","machine_control_simulation","data_logging"): reasons.append("TRACE_ORDER_INVALID")
        if any(record.run_id!=trace.run_id or record.correlation_id!=trace.correlation_id for record in trace.records): reasons.append("TRACE_IDENTITY_INVALID")
        if trace.execution_authorized or any(record.execution_authorized for record in trace.records): reasons.append("TRACE_EXECUTION_FORBIDDEN")
        record=trace.records[0]
        if record.status is not DryRunStatus.ACCEPTED or record.source_frame_id!=vision.frame_id or record.timestamp!=vision.frame_timestamp or record.clock_domain!=vision.frame_clock_domain or record.coordinate_frame!=vision.coordinate_frame or record.units!=vision.units: reasons.append("TRACE_SOURCE_IDENTITY_MISMATCH")
        if not trace.audit_complete: reasons.append("TRACE_AUDIT_INCOMPLETE")
        return reasons
    def _artifact(self,vision,trace,fixture,result,reasons,preview_id):
        verified=result is not None and result.status is VisionPathStatus.VERIFIED and result.solver_result is not None and result.solver_result.fully_covered and fixture==SQUARE_MITER_FIXTURE and np.array_equal(vision.left.points,CANONICAL_LEFT) and np.array_equal(vision.right.points,CANONICAL_RIGHT) and not reasons
        center=None if not verified else tuple(map(tuple,result.solver_result.segments[0].xy))
        diagnostics=() if result is None else tuple(result.solver_result.diagnostics.excluded if result.solver_result else ())
        return IsaacPreviewArtifact(PREVIEW_SCHEMA_VERSION,preview_id or uuid4().hex,trace.run_id,trace.correlation_id,vision.frame_id,vision.frame_timestamp,vision.frame_clock_domain,vision.coordinate_frame,vision.units,fixture,tuple(map(tuple,vision.left.points)),tuple(map(tuple,vision.right.points)),center,"SYNTHETIC_GEOMETRY_VERIFIED" if verified else "REJECTED",diagnostics,"POINT_PROVENANCE_AVAILABLE" if verified else "UNAVAILABLE",dict(DISPLAY_TRANSFORM),verified,False,tuple(dict.fromkeys(reasons or (("STRICT_GATE_B_BLOCKED",) if verified else ("PREVIEW_GEOMETRY_REJECTED",)))))
