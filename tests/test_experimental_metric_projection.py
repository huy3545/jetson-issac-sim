import unittest
import numpy as np

from data_source import CameraCalibration
from path_planning.experimental_metric_projection import (
    ExperimentalVisionMetricProjectionAdapter, MetricGeometryStatus,
    MetricUncertaintyStatus, TransformEvidence, TransformValidity,
)
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
                              TrackingState, VisionResult)


class MetricProjectionTests(unittest.TestCase):
    def setUp(self):
        self.points_l = np.array([[40., 20.], [40., 30.], [40., 40.]])
        self.points_r = np.array([[60., 20.], [60., 30.], [60., 40.]])
        self.depth = np.full((80, 100), .5)
        self.camera = CameraCalibration(100., 120., 50., 30., np.eye(4))
        self.matrix = np.eye(4)
        self.adapter = ExperimentalVisionMetricProjectionAdapter()

    def _vision(self, **changes):
        line = lambda points: LineVisionResult(points=points, observation_status=ObservationStatus.OBSERVED,
            tracking_state=TrackingState.TRACKING, verified=True,
            orientation_source=OrientationSource.REFERENCE,
            depth_valid=np.ones(len(points), dtype=bool))
        values = dict(frame_timestamp=10., frame_clock_domain="simulation", frame_id="frame-1",
            coordinate_frame="camera_optical", units="pixels", camera_to_reference=self.matrix,
            left=line(self.points_l), right=line(self.points_r))
        values.update(changes)
        return VisionResult(**values)

    def _evidence(self, matrix=None, **changes):
        values = dict(camera_frame="camera_optical", reference_frame="workpiece_m", units="m",
            intrinsics_id="intrinsic", intrinsics_version="v1", extrinsics_id="extrinsic",
            extrinsics_version="v1", calibration_source="synthetic fixture",
            transform_source="synthetic fixture", calibration_timestamp=9., transform_timestamp=9.,
            observation_timestamp=10., clock_domain="simulation", source_id="frame-1",
            transform_matrix=self.matrix if matrix is None else matrix,
            transform_validity=TransformValidity.SYNTHETIC)
        values.update(changes)
        return TransformEvidence(**values)

    def _expected(self, points, matrix, z=.5):
        camera = np.column_stack(((points[:, 0]-50)*z/100, (points[:, 1]-30)*z/120,
                                  np.full(len(points), z), np.ones(len(points))))
        return (camera @ matrix.T)[:, :3]

    def test_identity_translation_rotation_and_combined_match_analytic_projection(self):
        matrices = [np.eye(4), np.array([[1.,0.,0.,.1],[0.,1.,0.,-.2],[0.,0.,1.,.3],[0.,0.,0.,1.]]),
                    np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])]
        combined = matrices[2].copy(); combined[:3, 3] = [-.4, .25, .1]; matrices.append(combined)
        for matrix in matrices:
            vision = self._vision(camera_to_reference=matrix)
            result = self.adapter.project(vision, self.depth, self.camera, self._evidence(matrix))
            self.assertEqual(result.status, MetricGeometryStatus.UNCERTAIN)
            self.assertEqual(result.uncertainty_status, MetricUncertaintyStatus.UNKNOWN)
            np.testing.assert_allclose(result.segments[0].xyz, self._expected(self.points_l, matrix))
            self.assertFalse(result.execution_authorized)

    def test_multiple_depth_values_and_bilinear_sampling(self):
        depth = np.indices((80, 100))[1] * .01 + .2
        result = self.adapter.project(self._vision(), depth, self.camera, self._evidence())
        self.assertEqual(result.status, MetricGeometryStatus.UNCERTAIN)
        self.assertTrue(np.isfinite(result.segments[0].xyz).all())
        self.assertNotEqual(result.segments[0].xyz[0, 2], result.segments[1].xyz[0, 2])

    def test_invalid_depth_and_pixel_correspondence_reject(self):
        for depth in (np.zeros_like(self.depth), np.full_like(self.depth, np.nan)):
            result = self.adapter.project(self._vision(), depth, self.camera, self._evidence())
            self.assertEqual(result.status, MetricGeometryStatus.INVALID)
        bad = self._vision(left=LineVisionResult(points=self.points_l, observation_status=ObservationStatus.OBSERVED,
            tracking_state=TrackingState.TRACKING, verified=True, orientation_source=OrientationSource.REFERENCE,
            depth_valid=np.array([True, False, True])))
        result = self.adapter.project(bad, self.depth, self.camera, self._evidence())
        self.assertIn("LEFT_DEPTH_VALIDITY_UNPROVEN", result.diagnostics)

    def test_invalid_intrinsics_and_all_nonrigid_transforms_reject(self):
        invalid_camera = type("C", (), {"fx": 0., "fy": 1., "cx": 0., "cy": 0.})()
        self.assertIn("INVALID_INTRINSICS", self.adapter.project(self._vision(), self.depth, invalid_camera, self._evidence()).diagnostics)
        invalids = []
        reflection = np.eye(4); reflection[0, 0] = -1; invalids.append(reflection)
        scale = np.eye(4); scale[0, 0] = 2; invalids.append(scale)
        shear = np.eye(4); shear[0, 1] = .1; invalids.append(shear)
        homogeneous = np.eye(4); homogeneous[3, 0] = .1; invalids.append(homogeneous)
        for matrix in invalids:
            result = self.adapter.project(self._vision(camera_to_reference=matrix), self.depth, self.camera, self._evidence(matrix))
            self.assertEqual(result.status, MetricGeometryStatus.INVALID)

    def test_frame_time_source_and_missing_evidence_reject(self):
        cases = [
            (self._vision(coordinate_frame="other"), self._evidence()),
            (self._vision(frame_timestamp=11.), self._evidence()),
            (self._vision(frame_clock_domain="other"), self._evidence()),
            (self._vision(frame_id="other"), self._evidence()),
            (self._vision(), None),
        ]
        for vision, evidence in cases:
            self.assertEqual(self.adapter.project(vision, self.depth, self.camera, evidence).status, MetricGeometryStatus.INVALID)

    def test_predicted_expired_ambiguous_and_gap_observations_reject(self):
        predicted = LineVisionResult(points=self.points_l, observation_status=ObservationStatus.OCCLUDED,
            tracking_state=TrackingState.PREDICTING, verified=False, prediction_uncertainty_px=1.)
        for left in (predicted,
                     LineVisionResult(points=self.points_l, observation_status=ObservationStatus.OBSERVED,
                         tracking_state=TrackingState.TRACKING, verified=True, depth_valid=np.ones(3, bool), missing_ranges=((0, 1),)),
                     LineVisionResult(points=self.points_l, observation_status=ObservationStatus.OBSERVED,
                         tracking_state=TrackingState.TRACKING, verified=True, depth_valid=np.ones(3, bool), orientation_source=OrientationSource.AMBIGUOUS)):
            result = self.adapter.project(self._vision(left=left), self.depth, self.camera, self._evidence())
            self.assertEqual(result.status, MetricGeometryStatus.INVALID)

    def test_evidence_snapshot_and_output_are_immutable_and_never_verified(self):
        matrix = np.eye(4)
        evidence = self._evidence(matrix, transform_validity=TransformValidity.VERIFIED)
        result = self.adapter.project(self._vision(), self.depth, self.camera, evidence)
        matrix[0, 3] = 99.
        self.assertEqual(result.status, MetricGeometryStatus.UNCERTAIN)
        self.assertEqual(result.segments[0].status, MetricGeometryStatus.UNCERTAIN)
        self.assertFalse(result.transform_evidence.transform_matrix.flags.writeable)
        self.assertFalse(result.segments[0].xyz.flags.writeable)
        with self.assertRaises(ValueError): result.segments[0].xyz[0, 0] = 1.

    def test_no_legacy_or_execution_dependencies_are_imported(self):
        import path_planning.experimental_metric_projection as module
        with open(module.__file__, encoding="utf-8") as stream:
            source = stream.read()
        for forbidden in ("from path_planning.path_planner import", "from path_planning.legacy_geometry_gate import",
                          "from motion_planning", "from safety", "from machine_controller", "from data_logger"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
