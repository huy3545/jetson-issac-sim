"""Stage 7B-1A deterministic tests for production validation boundaries."""
import unittest

from integration.contracts import (ConfidenceStatus, DerivationMethod, PointProvenance,
    ProvenanceObservationStatus, UncertaintyStatus)
from integration.validation import CompatibilityValidator, ProvenanceResolver


def root(point_id="root", **overrides):
    data=dict(point_id=point_id,source_point_ids=(),source_run_id="run-a",source_frame_id="frame-a",segment_id="segment-a",
      observation_status=ProvenanceObservationStatus.OBSERVED,confidence_status=ConfidenceStatus.VERIFIED,confidence_value=.9,
      uncertainty_status=UncertaintyStatus.KNOWN,timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",
      calibration_id="cal-a",transform_id="tf-a",config_version="cfg-a",derivation_method=DerivationMethod.OBSERVATION)
    data.update(overrides);return PointProvenance(**data)
def derived(point_id="derived",sources=("root",),**overrides):
    data=dict(point_id=point_id,source_point_ids=sources,source_run_id="run-a",source_frame_id="frame-a",segment_id="segment-a",
      observation_status=ProvenanceObservationStatus.OBSERVED,confidence_status=ConfidenceStatus.VERIFIED,confidence_value=.8,
      uncertainty_status=UncertaintyStatus.KNOWN,timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",
      calibration_id="cal-a",transform_id="tf-a",config_version="cfg-a",derivation_method=DerivationMethod.MIDPOINT)
    data.update(overrides);return PointProvenance(**data)

class ValidationTests(unittest.TestCase):
    def test_observed_root_and_multilevel_derived_are_accepted(self):
        a=root("a"); b=derived("b",("a",)); c=derived("c",("b",))
        self.assertTrue(ProvenanceResolver({"a":a,"b":b,"c":c}).validate("c").accepted)

    def test_missing_predicted_unknown_cycle_and_cross_run_are_rejected(self):
        missing=derived(sources=("gone",));self.assertIn("SOURCE_PROVENANCE_UNRESOLVED",ProvenanceResolver({"derived":missing}).validate("derived").reason_codes)
        predicted=root("p",observation_status=ProvenanceObservationStatus.PREDICTED)
        self.assertIn("SOURCE_PREDICTED",ProvenanceResolver({"p":predicted,"derived":derived(sources=("p",))}).validate("derived").reason_codes)
        unknown=root("u",observation_status=ProvenanceObservationStatus.UNKNOWN)
        self.assertIn("SOURCE_OBSERVATION_UNKNOWN",ProvenanceResolver({"u":unknown,"derived":derived(sources=("u",))}).validate("derived").reason_codes)
        x=derived("x",("y",));y=derived("y",("x",));self.assertIn("PROVENANCE_CYCLE",ProvenanceResolver({"x":x,"y":y}).validate("x").reason_codes)
        other=root("o",source_run_id="run-b");self.assertIn("SOURCE_RUN_MISMATCH",ProvenanceResolver({"o":other,"derived":derived(sources=("o",))}).validate("derived").reason_codes)

    def test_compatibility_detects_all_explicit_mismatches_and_sync_policy(self):
        validator=CompatibilityValidator(); left=root(); right=root("right",source_frame_id="frame-b",timestamp=1.1,clock_domain="other",coordinate_frame="camera",units="px",calibration_id="cal-b",transform_id="tf-b",config_version="cfg-b",segment_id="segment-b",source_run_id="run-b")
        result=validator.compare(left,right,require_segment=True)
        for code in ("SOURCE_RUN_MISMATCH","FRAME_TIME_MISMATCH","CLOCK_DOMAIN_MISMATCH","COORDINATE_FRAME_MISMATCH","UNITS_MISMATCH","CALIBRATION_TRANSFORM_MISMATCH","CONFIG_VERSION_MISMATCH","SEGMENT_IDENTITY_MISMATCH"):self.assertIn(code,result.reason_codes)
        near=root("near",timestamp=1.01)
        self.assertFalse(validator.compare(left,near).accepted)
        self.assertTrue(validator.compare(left,near,timestamp_tolerance_s=.02).accepted)

    def test_contract_rejects_bool_numbers_and_malformed_nested_structures(self):
        with self.assertRaisesRegex(ValueError,"finite"):root(timestamp=True)
        with self.assertRaisesRegex(ValueError,"source_point_ids"):derived(sources=[])
        with self.assertRaisesRegex(ValueError,"observed point"):root(source_point_ids=("fake",))
        with self.assertRaisesRegex(ValueError,"tuple of non-empty strings"):
            from integration.contracts import IntegrationRecord,AdmissionStatus
            IntegrationRecord("run","corr","stage",AdmissionStatus.ACCEPTED,reason_codes=["bad"])


if __name__=="__main__":unittest.main()
