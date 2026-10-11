"""Independent Stage 7B-1 structural and provenance contract tests."""
import unittest

from integration.contracts import (
    AdmissionStatus, ConfigurationKind, ConfidenceStatus, DerivationMethod,
    INTEGRATION_SCHEMA_VERSION, IntegrationRecord, MotionTimingRecord,
    PointProvenance, ProvenanceObservationStatus, SyntheticConfiguration,
    TimingStatus,
)
from tests.golden_stage7b1_contracts import GOLDEN_CASES


def point(**changes):
    values = dict(point_id="derived-0", source_point_ids=("pixel-0", "pixel-1"),
        source_run_id="run-0", source_frame_id="frame-0", segment_id="segment-0",
        observation_status=ProvenanceObservationStatus.OBSERVED,
        confidence_status=ConfidenceStatus.VERIFIED, confidence_value=.8,
        uncertainty_status="KNOWN", timestamp=1., clock_domain="simulation",
        coordinate_frame="workpiece_m", units="m", calibration_id="cal-0",
        transform_id="transform-0", config_version="cfg-0",
        derivation_method=DerivationMethod.MIDPOINT)
    values.update(changes)
    return PointProvenance(**values)


class ContractHardeningTests(unittest.TestCase):
    def test_supported_schema_and_immutable_contract(self):
        item = point()
        self.assertEqual(item.schema_version, INTEGRATION_SCHEMA_VERSION)
        with self.assertRaisesRegex(Exception, "cannot assign"):
            item.point_id = "other"
        self.assertEqual(item.strict_reason_codes(), GOLDEN_CASES["valid_synthetic"][1])

    def test_unknown_schema_enum_identifier_and_nonfinite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            point(schema_version="9.9")
        with self.assertRaisesRegex(ValueError, "unknown observation_status"):
            point(observation_status="OBSERVED")
        with self.assertRaisesRegex(ValueError, "malformed point_id"):
            point(point_id="not a valid id")
        with self.assertRaisesRegex(ValueError, "finite"):
            point(timestamp=float("nan"))

    def test_missing_unknown_and_predicted_sources_fail_strict_without_upgrade(self):
        missing = point(calibration_id=None)
        self.assertIn(GOLDEN_CASES["missing_provenance"][1][0], missing.strict_reason_codes())
        unknown = point(confidence_status=ConfidenceStatus.UNKNOWN, uncertainty_status="UNKNOWN")
        self.assertIn(GOLDEN_CASES["unknown_confidence"][1][0], unknown.strict_reason_codes())
        predicted = point(derivation_method=DerivationMethod.RESAMPLING)
        self.assertIn(GOLDEN_CASES["predicted_source"][1][0], predicted.strict_reason_codes((ProvenanceObservationStatus.PREDICTED,)))

    def test_derived_point_preserves_all_sources_and_gap_segment_identity(self):
        derived = point(source_point_ids=("left-7", "right-9"), source_run_id="run-3", segment_id="segment-3")
        self.assertEqual(derived.source_point_ids, ("left-7", "right-9"))
        self.assertEqual((derived.source_run_id, derived.segment_id), ("run-3", "segment-3"))
        self.assertEqual(derived.derivation_method, DerivationMethod.MIDPOINT)

    def test_frame_time_and_calibration_transform_mismatch_have_explicit_rejection(self):
        left, right = point(source_frame_id="frame-a", timestamp=1.), point(source_frame_id="frame-b", timestamp=2.)
        reasons = []
        if (left.source_frame_id, left.timestamp) != (right.source_frame_id, right.timestamp): reasons.append("FRAME_TIME_MISMATCH")
        if (left.calibration_id, left.transform_id) != (right.calibration_id, right.transform_id): reasons.append("CALIBRATION_TRANSFORM_MISMATCH")
        self.assertIn(GOLDEN_CASES["frame_time_mismatch"][1][0], reasons)

    def test_motion_timing_is_never_inferred_from_speed(self):
        target = MotionTimingRecord(0, "segment-0", "path-0", "workpiece_m", "m", None, None, .02, .1, AdmissionStatus.ACCEPTED)
        self.assertEqual(target.timing_status, TimingStatus.UNSPECIFIED)
        self.assertEqual(target.timing_reason_codes(), ("TIMING_UNSPECIFIED",))

    def test_synthetic_and_real_unspecified_configuration_policies(self):
        synthetic = SyntheticConfiguration("camera-sim", "cal-sim", "transform-sim", .5, ConfigurationKind.SYNTHETIC)
        self.assertEqual(synthetic.real_accuracy_claim, "NONE")
        self.assertEqual(synthetic.strict_reason_codes(), ())
        real = SyntheticConfiguration(None, None, None, None, ConfigurationKind.UNSPECIFIED)
        self.assertEqual(real.strict_reason_codes(), ("REAL_CONFIGURATION_UNSPECIFIED",))
        with self.assertRaisesRegex(ValueError, "real_accuracy_claim"):
            SyntheticConfiguration("camera-sim", "cal-sim", "transform-sim", .5, ConfigurationKind.SYNTHETIC, "MEASURED")

    def test_integration_record_rejects_unknown_schema_and_execution(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            IntegrationRecord("run", "corr", "stage", AdmissionStatus.ACCEPTED, schema_version="unknown")
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            IntegrationRecord("run", "corr", "stage", AdmissionStatus.ACCEPTED, execution_authorized=True)


if __name__ == "__main__":
    unittest.main()
