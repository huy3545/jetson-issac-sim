import hashlib
import unittest
import numpy as np

from data_source import CameraCalibration
from path_planning.calibration_evidence import (
    CalibrationEvidence, CalibrationEvidenceValidator, CalibrationTrust,
    CalibrationTrustedProjectionAdapter,
)
from path_planning.experimental_metric_projection import TransformEvidence, TransformValidity
from vision.contracts import LineVisionResult, ObservationStatus, OrientationSource, TrackingState, VisionResult


class CalibrationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.artifact = b"synthetic calibration artifact v1"
        self.matrix = np.eye(4)
        self.camera = CameraCalibration(100., 100., 50., 30., self.matrix)
        line = lambda x: LineVisionResult(points=np.array([[x, 20.], [x, 30.]]),
            observation_status=ObservationStatus.OBSERVED, tracking_state=TrackingState.TRACKING,
            verified=True, orientation_source=OrientationSource.REFERENCE, depth_valid=np.ones(2, bool))
        self.vision = VisionResult(frame_timestamp=10., frame_clock_domain="sim", frame_id="frame-1",
            coordinate_frame="camera", units="pixels", camera_to_reference=self.matrix,
            left=line(40.), right=line(60.))
        self.transform = TransformEvidence("camera", "workpiece_m", "m", "i", "v1", "e", "v1",
            "fixture", "fixture", 9., 9., 10., "sim", "frame-1", self.matrix, TransformValidity.SYNTHETIC)

    def _evidence(self, **changes):
        values = dict(calibration_id="cal-1", calibration_version="v1", camera_id="cam-1",
            camera_frame="camera", reference_frame="workpiece_m", intrinsics_id="i", intrinsics_version="v1",
            extrinsics_id="e", extrinsics_version="v1", calibration_source="fixture",
            calibration_method="synthetic fixture", calibration_timestamp=9., clock_domain="sim", units="m",
            source_id="frame-1", declared_trust=CalibrationTrust.DECLARED)
        values.update(changes)
        return CalibrationEvidence(**values)

    def test_synthetic_and_declared_never_upgrade_to_sensor_verified(self):
        validator = CalibrationEvidenceValidator()
        synthetic = validator.validate(self._evidence(declared_trust=CalibrationTrust.SYNTHETIC))
        declared = validator.validate(self._evidence())
        self.assertEqual(synthetic.trust, CalibrationTrust.SYNTHETIC)
        self.assertEqual(declared.trust, CalibrationTrust.DECLARED)
        result = CalibrationTrustedProjectionAdapter().project(self.vision, np.full((80, 100), .5), self.camera,
            self.transform, synthetic.evidence)
        self.assertEqual(result.calibration.trust, CalibrationTrust.SYNTHETIC)
        self.assertEqual(result.metric_result.status.value, "UNCERTAIN")
        self.assertFalse(result.execution_authorized)

    def test_only_explicit_artifact_bytes_and_validation_claim_can_be_evidence_validated(self):
        digest = hashlib.sha256(self.artifact).hexdigest()
        without_accuracy_claim = self._evidence(calibration_artifact_reference="memory://fixture",
            calibration_artifact_digest=digest)
        self.assertEqual(CalibrationEvidenceValidator().validate(without_accuracy_claim, self.artifact).trust,
                         CalibrationTrust.DECLARED)
        evidence = self._evidence(calibration_artifact_reference="memory://fixture", calibration_artifact_digest=digest,
            validation_method="artifact integrity review", validation_timestamp=10., validation_result=True,
            declared_trust=CalibrationTrust.EVIDENCE_VALIDATED)
        validator = CalibrationEvidenceValidator()
        self.assertEqual(validator.validate(evidence).trust, CalibrationTrust.DECLARED)
        self.assertEqual(validator.validate(evidence, self.artifact).trust, CalibrationTrust.EVIDENCE_VALIDATED)

    def test_missing_fields_digest_and_claim_failures_are_invalid(self):
        validator = CalibrationEvidenceValidator()
        cases = [
            self._evidence(calibration_id=""),
            self._evidence(calibration_artifact_reference="memory://only-ref"),
            self._evidence(calibration_artifact_reference="memory://x", calibration_artifact_digest="x"),
            self._evidence(validation_result=True),
            self._evidence(validation_result=False),
        ]
        for evidence in cases:
            self.assertEqual(validator.validate(evidence).trust, CalibrationTrust.INVALID)
        digest = hashlib.sha256(self.artifact).hexdigest()
        evidence = self._evidence(calibration_artifact_reference="memory://x", calibration_artifact_digest=digest)
        self.assertEqual(validator.validate(evidence, b"changed").trust, CalibrationTrust.INVALID)

    def test_transform_consistency_mismatches_block_projection(self):
        adapter = CalibrationTrustedProjectionAdapter()
        for evidence in (self._evidence(camera_frame="wrong"), self._evidence(intrinsics_version="wrong"),
                         self._evidence(source_id="wrong"), self._evidence(clock_domain="other")):
            result = adapter.project(self.vision, np.full((80, 100), .5), self.camera, self.transform, evidence)
            self.assertEqual(result.calibration.trust, CalibrationTrust.INVALID)
            self.assertIsNone(result.metric_result)

    def test_synthetic_cannot_be_combined_with_non_synthetic_transform(self):
        transform = TransformEvidence("camera", "workpiece_m", "m", "i", "v1", "e", "v1", "fixture", "fixture",
            9., 9., 10., "sim", "frame-1", self.matrix, TransformValidity.DECLARED)
        result = CalibrationTrustedProjectionAdapter().project(self.vision, np.full((80, 100), .5), self.camera,
            transform, self._evidence(declared_trust=CalibrationTrust.SYNTHETIC))
        self.assertEqual(result.calibration.trust, CalibrationTrust.INVALID)
        self.assertIn("SYNTHETIC_TRUST_MISMATCH", result.calibration.diagnostics)

    def test_immutable_evidence_and_no_execution_dependencies(self):
        evidence = self._evidence(validation_diagnostics=("fixture",))
        result = CalibrationEvidenceValidator().validate(evidence)
        self.assertEqual(result.evidence.validation_diagnostics, ("fixture",))
        self.assertFalse(CalibrationTrustedProjectionAdapter().project(self.vision, np.full((80, 100), .5), self.camera,
            self.transform, evidence).execution_authorized)
        import path_planning.calibration_evidence as module
        with open(module.__file__, encoding="utf-8") as stream: source = stream.read()
        for forbidden in ("from motion_planning", "from safety", "from machine_controller", "from data_logger", "from .path_planner"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
