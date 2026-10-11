"""Stage 7C-2 provenance/uncertainty contract tests using synthetic evidence only."""
from dataclasses import replace
import unittest
import numpy as np

from data_source import CameraCalibration
from integration.contracts import AdmissionStatus
from integration.metric_bridge import MetricBridgeVerifier, MetricGeometryBridge
from path_planning.experimental_metric_projection import (
    ExperimentalVisionMetricProjectionAdapter, MetricGeometrySegment,
    MetricGeometryStatus, MetricUncertainty, MetricUncertaintyStatus,
    TransformEvidence, TransformValidity,
)
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
    PointDerivationMethod, TrackingState, VisionPointProvenance, VisionResult)


class PointLineageTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.eye(4)
        self.points = (np.array([[1.25, 1.5], [2.5, 2.25]]),
                       np.array([[4.25, 1.5], [5.5, 2.25]]))
        self.depth = np.full((8, 8), 0.5)
        self.camera = CameraCalibration(10., 10., 0., 0., self.matrix)
        self.evidence = TransformEvidence("camera", "workpiece_m", "m", "intrinsics", "v1",
            "transform", "v1", "synthetic", "synthetic", 0., 0., 2., "sim", "frame", self.matrix,
            TransformValidity.SYNTHETIC)

    def _line(self, name, points, status=ObservationStatus.OBSERVED, provenance=True):
        lineage = None if not provenance else tuple(VisionPointProvenance(
            name, index, tuple(point), status, TrackingState.TRACKING,
            PointDerivationMethod.DIRECT, lineage_available=True, timestamp=2., clock_domain="sim")
            for index, point in enumerate(points))
        return LineVisionResult(points=points, observation_status=status,
            tracking_state=TrackingState.TRACKING, confidence=.7, verified=True,
            orientation_source=OrientationSource.REFERENCE, depth_valid=np.ones(len(points), bool),
            point_provenance=lineage)

    def _vision(self, left_status=ObservationStatus.OBSERVED, provenance=True):
        return VisionResult(2., "sim", frame_id="frame", coordinate_frame="camera", units="pixels",
            camera_to_reference=self.matrix, left=self._line("left", self.points[0], left_status, provenance),
            right=self._line("right", self.points[1], provenance=provenance))

    def _result(self, **changes):
        vision = self._vision(**changes)
        return vision, ExperimentalVisionMetricProjectionAdapter().project(vision, self.depth, self.camera, self.evidence)

    def test_observed_pixel_depth_xyz_lineage_and_independent_verifier(self):
        vision, metric = self._result()
        self.assertEqual(metric.status, MetricGeometryStatus.UNCERTAIN)
        self.assertEqual(metric.uncertainty.xyz_status, MetricUncertaintyStatus.UNKNOWN)
        point = metric.segments[0].point_provenance[0]
        self.assertEqual(point.vision_point.source_index, 0)
        self.assertEqual(point.pixel_xy, tuple(self.points[0][0]))
        self.assertEqual(len(point.depth_samples), 4)
        self.assertAlmostEqual(sum(weight for _, _, weight in point.depth_samples), 1.)
        bridge = MetricGeometryBridge().convert(vision, metric, run_id="run", correlation_id="corr")
        self.assertEqual(bridge.admission_status, AdmissionStatus.BLOCKED)
        self.assertEqual(MetricBridgeVerifier().verify_lineage(vision, metric, bridge), ())

    def test_unavailable_lineage_is_disclosed_and_strict_remains_blocked(self):
        vision, metric = self._result(provenance=False)
        bridge = MetricGeometryBridge().convert(vision, metric, run_id="run", correlation_id="corr")
        self.assertIn("POINT_PROVENANCE_UNAVAILABLE", bridge.reason_codes)
        self.assertEqual(bridge.admission_status, AdmissionStatus.BLOCKED)
        self.assertIn("POINT_PROVENANCE_UNAVAILABLE", MetricBridgeVerifier().verify_lineage(vision, metric, bridge))

    def test_corrupted_point_lineage_and_metadata_are_detected(self):
        vision, metric = self._result()
        bridge = MetricGeometryBridge().convert(vision, metric, run_id="run", correlation_id="corr")
        bad_point = replace(bridge.segments[0].point_provenance[0], pixel_xy=(99., 99.))
        bad_segment = replace(bridge.segments[0], point_provenance=(bad_point,) + bridge.segments[0].point_provenance[1:])
        bad_bridge = replace(bridge, segments=(bad_segment,) + bridge.segments[1:])
        reasons = MetricBridgeVerifier().verify_lineage(vision, metric, bad_bridge)
        self.assertIn("POINT_LINEAGE_MISMATCH", reasons)
        self.assertIn("PIXEL_COORDINATE_MISMATCH", reasons)
        self.assertIn("SOURCE_ID_MISMATCH", MetricBridgeVerifier().verify_lineage(vision, replace(metric, source_id="other"), bridge))

    def test_depth_and_projection_admission_fail_closed(self):
        vision = self._vision()
        bad_depth = self.depth.copy(); bad_depth[1, 1] = np.nan
        result = ExperimentalVisionMetricProjectionAdapter().project(vision, bad_depth, self.camera, self.evidence)
        self.assertEqual(result.status, MetricGeometryStatus.INVALID)
        bad_intrinsics = type("Camera", (), {"fx": 0., "fy": 1., "cx": 0., "cy": 0.})()
        self.assertEqual(ExperimentalVisionMetricProjectionAdapter().project(vision, self.depth, bad_intrinsics, self.evidence).status, MetricGeometryStatus.INVALID)

    def test_occluded_is_not_rewritten_as_predicted_or_observed(self):
        vision, metric = self._result()
        occluded = replace(vision, left=self._line("left", self.points[0], ObservationStatus.OCCLUDED))
        altered = replace(metric, segments=(replace(metric.segments[0], point_provenance=metric.segments[0].point_provenance), metric.segments[1]))
        bridge = MetricGeometryBridge().convert(occluded, altered, run_id="run", correlation_id="corr")
        self.assertEqual(bridge.segments[0].observation_status, "OCCLUDED")
        self.assertIn("NON_OBSERVED_SOURCE", bridge.reason_codes)
        self.assertNotIn("PREDICTED_SOURCE", bridge.reason_codes)

    def test_uncertainty_and_snapshots_cannot_claim_real_calibration(self):
        vision, metric = self._result()
        self.assertEqual(metric.uncertainty.evidence_kind, "UNAVAILABLE")
        self.assertFalse(metric.execution_authorized)
        self.assertFalse(metric.segments[0].xyz.flags.writeable)
        with self.assertRaises(ValueError): metric.segments[0].xyz.setflags(write=True)
        self.assertEqual(metric.segments[0].uncertainty.xyz_status, MetricUncertaintyStatus.UNKNOWN)

    def test_unknown_schema_and_controller_dependency_are_rejected_or_absent(self):
        with self.assertRaises(ValueError): VisionPointProvenance("left", 0, (1., 1.), ObservationStatus.OBSERVED,
            TrackingState.TRACKING, PointDerivationMethod.DIRECT, schema_version="unknown")
        import integration.metric_bridge as bridge_module
        with open(bridge_module.__file__, encoding="utf-8") as source:
            self.assertNotIn("controller", source.read().lower())


if __name__ == "__main__":
    unittest.main()
