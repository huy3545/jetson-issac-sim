"""Stage 7B contract integration tests; no controller or hardware calls."""
import unittest
import numpy as np

from config import Config
from data_source import CameraCalibration
from integration.experimental_harness import ExperimentalIntegrationHarness, InMemoryAuditSink
from path_planning.path_qualification import (PathQualificationStatus, QualifiedPathResult,
    QualifiedPathSegment, UncertaintyStatus)
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
    TrackingState, VisionResult)


def vision(**changes):
    matrix = np.eye(4)
    line = LineVisionResult(np.array([[.0, .0], [1., .0]]), ObservationStatus.OBSERVED,
        TrackingState.TRACKING, .9, depth_valid=np.ones(2, dtype=bool),
        orientation_source=OrientationSource.REFERENCE, verified=True)
    values = dict(frame_timestamp=1., frame_clock_domain="simulation", frame_id="frame-1",
        coordinate_frame="workpiece_m", units="m", camera_to_reference=matrix,
        left=line, right=LineVisionResult(np.array([[.0, .1], [1., .1]]), ObservationStatus.OBSERVED,
        TrackingState.TRACKING, .9, depth_valid=np.ones(2, dtype=bool),
        orientation_source=OrientationSource.REFERENCE, verified=True))
    values.update(changes)
    return VisionResult(**values)


def qualified(*, uncertainty=UncertaintyStatus.KNOWN, status=PathQualificationStatus.VERIFIED, count=1):
    xyz = np.array([[0., 0., .5], [.1, 0., .5], [.2, 0., .5]])
    s = np.array([0., .1, .2]); tangent = np.tile([1., 0., 0.], (3, 1))
    segments = tuple(QualifiedPathSegment(f"s:{i}", 1, i, (0, 3), xyz, s, tangent, np.zeros(3),
        uncertainty, status, "workpiece_m", "m", 1., "simulation") for i in range(count))
    return QualifiedPathResult("1.1", segments)


class FailingSink:
    def emit(self, event):
        raise OSError("audit storage unavailable")


class ExperimentalIntegrationTests(unittest.TestCase):
    def execute_harness(self, item=vision(), path=qualified(), sink=None):
        return ExperimentalIntegrationHarness(Config(), sink).run(item, path, run_id="run-1", correlation_id="corr-1")

    def test_contract_propagation_and_no_execution_authorization(self):
        result = self.execute_harness(); records = {item.stage_id: item for item in result.records}
        self.assertEqual(records["vision"].admission_status.value, "ACCEPTED")
        self.assertEqual(records["motion"].admission_status.value, "ACCEPTED")
        self.assertEqual(records["motion"].segment_id, "s:0")
        self.assertEqual(records["motion"].config_version, Config().config_version)
        self.assertFalse(result.execution_authorized)
        self.assertTrue(all(not record.execution_authorized for record in result.records))

    def test_predicted_missing_provenance_and_ambiguous_geometry_reject(self):
        predicted = LineVisionResult(np.array([[0., 0.], [1., 0.]]), ObservationStatus.OCCLUDED,
            TrackingState.PREDICTING, .1, prediction_uncertainty_px=2., verified=False)
        bad = vision(left=predicted, coordinate_frame="", units="pixels", frame_id=None)
        records = {item.stage_id: item for item in self.execute_harness(bad).records}
        self.assertEqual(records["vision"].admission_status.value, "REJECTED")
        self.assertIn("LEFT_NOT_OBSERVED_VERIFIED", records["vision"].reason_codes)
        self.assertEqual(records["path"].admission_status.value, "BLOCKED")

    def test_metric_provenance_and_ambiguous_path_are_rejected(self):
        missing = vision(coordinate_frame="", units="pixels")
        records = {item.stage_id: item for item in self.execute_harness(missing).records}
        self.assertEqual(records["path"].admission_status.value, "REJECTED")
        self.assertIn("UNITS_NOT_METRIC", records["path"].reason_codes)
        ambiguous = LineVisionResult(np.array([[0., 0.], [1., 0.]]), ObservationStatus.OBSERVED,
            TrackingState.TRACKING, .9, depth_valid=np.ones(2, dtype=bool),
            orientation_source=OrientationSource.AMBIGUOUS, verified=True)
        records = {item.stage_id: item for item in self.execute_harness(vision(left=ambiguous)).records}
        self.assertEqual(records["path"].admission_status.value, "REJECTED")
        self.assertIn("AMBIGUOUS_IDENTITY", records["path"].reason_codes)

    def test_unknown_uncertainty_gap_and_motion_rejection_propagate(self):
        result = self.execute_harness(path=qualified(uncertainty=UncertaintyStatus.UNKNOWN, count=2))
        records = {item.stage_id: item for item in result.records}
        self.assertEqual(records["motion"].admission_status.value, "REJECTED")
        self.assertIn("UNCERTAINTY_NOT_VERIFIED", records["motion"].reason_codes)
        self.assertIn("GAPPED_SEGMENT", records["safety_assessment"].reason_codes)
        self.assertEqual(records["safety_assessment"].admission_status.value, "REJECTED")

    def test_time_mismatch_and_rejection_events_keep_correlation(self):
        bad = vision(time_sync_valid=False)
        result = self.execute_harness(bad, path=None)
        self.assertEqual(result.final_status.value, "BLOCKED")
        self.assertTrue(any("TIME_SYNC_INVALID" in event.reason_codes for event in result.events))
        self.assertTrue(all(event.run_id == "run-1" and event.correlation_id == "corr-1" for event in result.events))

    def test_safety_independently_fails_closed_on_missing_machine_state(self):
        records = {item.stage_id: item for item in self.execute_harness().records}
        self.assertIn("MACHINE_STATE_EVIDENCE_MISSING", records["safety_assessment"].reason_codes)
        self.assertIn("EXECUTION_NOT_AUTHORIZED", records["safety_assessment"].reason_codes)

    def test_structured_logging_and_logging_failure_are_isolated(self):
        sink = InMemoryAuditSink(); success = self.execute_harness(sink=sink)
        self.assertTrue(success.audit_complete); self.assertEqual(len(sink.events), len(success.events))
        failed = self.execute_harness(sink=FailingSink())
        self.assertFalse(failed.audit_complete)
        self.assertTrue(any(event.stage_id == "audit_logging" for event in failed.events))

    def test_no_controller_import_or_invocation(self):
        import integration.experimental_harness as module
        self.execute_harness()
        self.assertNotIn("controller.machine_controller", module.__dict__)


if __name__ == "__main__":
    unittest.main()
