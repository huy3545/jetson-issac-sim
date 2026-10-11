"""Stage 7D-1: six-module passive integration trace."""
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

from integration.six_module_dry_run import (AuditWriteStatus, DryRunStatus,
                                             InMemorySixModuleAuditSink, SixModuleDryRunHarness)
from path_planning.planar_medial_axis import (PlanarMedialAxisConfig, SolverDiagnostics,
                                               SolverResult, SolverStatus)
from path_planning.vision_adapter import VisionPathAdapter
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
                              PointDerivationMethod, TrackingState,
                              VisionPointProvenance, VisionResult)


class FailingSink:
    def emit(self, record): raise RuntimeError("sink unavailable")


class FailingAfterSink:
    def __init__(self, successful_writes): self.successful_writes=successful_writes; self.records=[]
    def emit(self, record):
        if len(self.records) >= self.successful_writes: raise RuntimeError("chosen write failure")
        self.records.append(record)


class UnknownGeometrySolver:
    def __init__(self): self.calls=0; self.config=PlanarMedialAxisConfig()
    def solve(self, left, right):
        self.calls += 1
        return SolverResult((),SolverStatus.INVALID_GEOMETRY,False,
                            SolverDiagnostics(excluded=("NEW_GEOMETRY_FAILURE_CODE",)))


class SixModuleDryRunTests(unittest.TestCase):
    def _line(self,name,points,*,provenance=True,status=ObservationStatus.OBSERVED,gaps=()):
        lineage=None if not provenance else tuple(VisionPointProvenance(name,index,tuple(point),status,
            TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,
            timestamp=3.,clock_domain="sim") for index,point in enumerate(points))
        return LineVisionResult(np.asarray(points,float),status,TrackingState.TRACKING,.9,
            depth_valid=np.ones(len(points),bool),orientation_source=OrientationSource.REFERENCE,
            verified=True,point_provenance=lineage,missing_ranges=gaps)
    def _vision(self,**changes):
        left=np.array([[0.,.1],[.5,.1],[.5,.6]])
        right=np.array([[0.,-.1],[.7,-.1],[.7,.6]])
        values=dict(frame_timestamp=3.,frame_clock_domain="sim",frame_id="frame-7d1",
            coordinate_frame="workpiece_m",units="m",left=self._line("left",left),right=self._line("right",right))
        values.update(changes); return VisionResult(**values)
    def _run(self,vision=None,**kwargs):
        return SixModuleDryRunHarness(**kwargs.pop("harness",{})).run(vision or self._vision(),run_id="run-7d1",correlation_id="corr-7d1",**kwargs)

    def test_synthetic_square_trace_is_ordered_and_gate_b_stays_blocked(self):
        result=self._run()
        self.assertEqual(tuple(record.stage_id for record in result.records),SixModuleDryRunHarness.ORDER)
        self.assertTrue(all(record.run_id=="run-7d1" and record.correlation_id=="corr-7d1" for record in result.records))
        path=result.records[1]; self.assertEqual(path.geometry_status,"SYNTHETIC_GEOMETRY_VERIFIED")
        self.assertEqual(path.status,DryRunStatus.BLOCKED)
        self.assertIn("STRICT_GATE_B_BLOCKED",path.reason_codes)
        self.assertIn("CALIBRATION_UNVERIFIED",path.reason_codes)
        self.assertIn("XYZ_UNCERTAINTY_UNKNOWN",path.reason_codes)
        self.assertIsNone(path.source_path_id)
        self.assertEqual(path.source_segment_ids,())
        self.assertEqual(result.final_status,DryRunStatus.BLOCKED); self.assertTrue(result.audit_complete)
        self.assertTrue(all(not record.execution_authorized for record in result.records))

    def test_blocked_motion_safety_control_and_logging_are_explicit(self):
        sink=InMemorySixModuleAuditSink(); result=self._run(harness={"audit_sink":sink})
        motion,safety,control,logging=result.records[2:]
        self.assertIn("UPSTREAM_STRICT_ADMISSION_BLOCKED",motion.reason_codes)
        self.assertIn("UPSTREAM_STRICT_ADMISSION_BLOCKED",safety.reason_codes)
        self.assertIn("UPSTREAM_STRICT_ADMISSION_BLOCKED",control.reason_codes)
        self.assertEqual(logging.status,DryRunStatus.ACCEPTED)
        self.assertEqual(len(sink.records),6)
        self.assertFalse(result.execution_authorized)

    def test_missing_provenance_and_observation_failures_are_preserved(self):
        vision=self._vision(left=self._line("left",np.array([[0.,.1],[.5,.1],[.5,.6]]),provenance=False))
        result=self._run(vision)
        self.assertEqual(result.records[0].status,DryRunStatus.REJECTED)
        self.assertIn("LEFT_POINT_PROVENANCE_UNAVAILABLE",result.records[0].reason_codes)
        self.assertEqual(result.records[1].status,DryRunStatus.SKIPPED)
        occluded=self._run(self._vision(left=self._line("left",np.array([[0.,.1],[.5,.1],[.5,.6]]),status=ObservationStatus.OCCLUDED)))
        self.assertIn("LEFT_NOT_OBSERVED",occluded.records[0].reason_codes)

    def test_identity_time_calibration_and_uncertainty_mismatches_do_not_upgrade(self):
        result=self._run(expected_frame_id="other",expected_timestamp=4.,expected_clock_domain="other",calibration_id=None,transform_id=None,uncertainty_status="UNKNOWN")
        reasons=result.records[0].reason_codes
        self.assertIn("SOURCE_FRAME_ID_MISMATCH",reasons); self.assertIn("TIMESTAMP_MISMATCH",reasons); self.assertIn("CLOCK_DOMAIN_MISMATCH",reasons)
        # Synthetic geometry remains locally visible but missing calibration and
        # UNKNOWN uncertainty cannot become a strict admission.
        normal=self._run(calibration_id=None,transform_id=None,uncertainty_status="UNKNOWN")
        self.assertEqual(normal.records[1].status,DryRunStatus.BLOCKED)
        self.assertEqual(normal.records[1].uncertainty_status,"UNKNOWN")

    def test_reflex_turnback_and_fatal_geometry_are_not_promoted(self):
        reflex=self._run(self._vision(right=self._line("right",np.array([[0.,-.1],[.5,-.1],[.5,-.2]]))))
        self.assertEqual(reflex.records[1].status,DryRunStatus.REJECTED)
        self.assertIn("REFLEX_ORIENTATION_UNPROVEN",reflex.records[1].reason_codes)
        crossing=self._run(self._vision(left=self._line("left",np.array([[0.,.1],[1.,-.1]])),right=self._line("right",np.array([[0.,-.1],[1.,.1]]))))
        self.assertIn("BOUNDARY_INTERSECTION",crossing.records[1].reason_codes)

    def test_audit_failure_is_visible_and_not_a_success(self):
        result=self._run(harness={"audit_sink":FailingSink()})
        self.assertFalse(result.audit_complete); self.assertEqual(result.final_status,DryRunStatus.ERROR)
        self.assertEqual(result.records[-1].status,DryRunStatus.ERROR)
        self.assertIn("AUDIT_LOGGING_FAILED",result.records[-1].reason_codes)
        self.assertEqual(result.audit_write_status,AuditWriteStatus.ZERO_ACKNOWLEDGED)
        self.assertEqual(result.records_confirmed,0)

    def test_declared_metadata_never_upgrades_evidence(self):
        result=self._run(calibration_id="declared-cal",transform_id="declared-tf",uncertainty_status="KNOWN")
        path=result.records[1]
        self.assertEqual(path.calibration_evidence_status,"DECLARED_UNVERIFIED")
        self.assertEqual(path.transform_evidence_status,"DECLARED_UNVERIFIED")
        self.assertEqual(path.uncertainty_evidence_status,"DECLARED_UNVERIFIED")
        self.assertEqual(path.status,DryRunStatus.BLOCKED)
        malformed=self._run(calibration_id="",transform_id=4,uncertainty_status="MAGIC")
        self.assertEqual(malformed.records[0].status,DryRunStatus.REJECTED)
        self.assertIn("INVALID_CALIBRATION_ID",malformed.records[0].reason_codes)
        self.assertIn("INVALID_TRANSFORM_ID",malformed.records[0].reason_codes)

    def test_adapter_preflight_and_path_rejection_propagate_truthfully(self):
        bad_tracking=self._run(self._vision(left=replace(self._vision().left,tracking_state=TrackingState.PREDICTING)))
        self.assertIn("LEFT_NOT_TRACKING",bad_tracking.records[0].reason_codes)
        bad_depth=self._run(self._vision(left=replace(self._vision().left,depth_valid=np.array([True,False,True]))))
        self.assertIn("LEFT_DEPTH_INVALID",bad_depth.records[0].reason_codes)
        ambiguous=self._run(self._vision(left=replace(self._vision().left,orientation_source=OrientationSource.AMBIGUOUS)))
        self.assertIn("AMBIGUOUS_IDENTITY",ambiguous.records[0].reason_codes)
        reflex=self._run(self._vision(right=self._line("right",np.array([[0.,-.1],[.5,-.1],[.5,-.2]]))))
        self.assertEqual(reflex.records[1].status,DryRunStatus.REJECTED)
        self.assertEqual(reflex.records[2].status,DryRunStatus.SKIPPED)
        self.assertIn("UPSTREAM_REJECTED",reflex.records[2].reason_codes)
        self.assertEqual(reflex.final_status,DryRunStatus.REJECTED)
        self.assertIn("REFLEX_ORIENTATION_UNPROVEN",reflex.records[1].diagnostics)

    def test_audit_acknowledgement_is_not_durability_and_partial_is_explicit(self):
        complete=self._run()
        self.assertEqual(complete.audit_write_status,AuditWriteStatus.COMPLETE_ACKNOWLEDGED)
        self.assertEqual(complete.records_confirmed,6)
        self.assertEqual(complete.records[-1].status,DryRunStatus.ACCEPTED)
        self.assertEqual(complete.final_status,DryRunStatus.BLOCKED)
        for count,status in ((1,AuditWriteStatus.PARTIAL_ACKNOWLEDGED),(4,AuditWriteStatus.PARTIAL_ACKNOWLEDGED)):
            result=self._run(harness={"audit_sink":FailingAfterSink(count)})
            self.assertFalse(result.audit_complete); self.assertEqual(result.records_confirmed,count)
            self.assertEqual(result.audit_write_status,status); self.assertEqual(result.final_status,DryRunStatus.ERROR)
            self.assertEqual(len(result.records),6)

    def test_no_controller_or_motion_targets_are_exposed(self):
        result=self._run()
        self.assertFalse(result.execution_authorized)
        self.assertTrue(all(not item.execution_authorized for item in result.records))
        self.assertNotIn("import MachineController",Path("integration/six_module_dry_run.py").read_text())

    def test_invalid_vision_keeps_a_six_stage_error_trace(self):
        result=SixModuleDryRunHarness().run(object(),run_id="bad-run",correlation_id="bad-corr")
        self.assertEqual(tuple(record.stage_id for record in result.records),SixModuleDryRunHarness.ORDER)
        self.assertEqual(result.records[0].status,DryRunStatus.ERROR)
        self.assertEqual(result.final_status,DryRunStatus.ERROR)
        self.assertEqual(result.records_confirmed,6)

    def test_public_preflight_separates_observation_from_geometry(self):
        adapter=VisionPathAdapter(require_point_provenance=True)
        self.assertEqual(adapter.validate_observation(self._vision()),())
        invalid=self._vision(left=replace(self._vision().left,tracking_state=TrackingState.LOST))
        self.assertIn("LEFT_NOT_TRACKING",adapter.validate_observation(invalid))
        solver=UnknownGeometrySolver()
        harness=SixModuleDryRunHarness()
        harness.path_adapter=VisionPathAdapter(solver=solver,require_point_provenance=True)
        result=harness.run(self._vision(),run_id="geometry-run",correlation_id="geometry-corr")
        self.assertEqual(solver.calls,1)
        self.assertEqual(tuple(item.stage_id for item in result.records),SixModuleDryRunHarness.ORDER)
        self.assertEqual(result.records[0].status,DryRunStatus.ACCEPTED)
        self.assertNotIn("NEW_GEOMETRY_FAILURE_CODE",result.records[0].reason_codes)
        self.assertEqual(result.records[1].status,DryRunStatus.REJECTED)
        self.assertIn("NEW_GEOMETRY_FAILURE_CODE",result.records[1].reason_codes)
        self.assertEqual(result.records[2].status,DryRunStatus.SKIPPED)
        self.assertEqual(result.records[-1].status,DryRunStatus.ACCEPTED)
        self.assertEqual(result.final_status,DryRunStatus.REJECTED)
        self.assertTrue(all(not item.execution_authorized for item in result.records))

    def test_invalid_preflight_does_not_invoke_solver(self):
        solver=UnknownGeometrySolver(); harness=SixModuleDryRunHarness()
        harness.path_adapter=VisionPathAdapter(solver=solver,require_point_provenance=True)
        invalid=self._vision(left=replace(self._vision().left,tracking_state=TrackingState.PREDICTING))
        result=harness.run(invalid,run_id="invalid-run",correlation_id="invalid-corr")
        self.assertEqual(solver.calls,0)
        self.assertEqual(result.records[0].status,DryRunStatus.REJECTED)
        self.assertEqual(result.records[1].status,DryRunStatus.SKIPPED)


if __name__=="__main__": unittest.main()
