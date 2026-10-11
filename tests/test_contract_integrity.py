"""Stage 7B-1b production invariants and targeted manual mutation checks."""
import unittest

from integration.contracts import (AdmissionStatus, ConfidenceStatus, DerivationMethod,
    IntegrationEvent, IntegrationRecord, IntegrationRunResult, MotionTimingRecord,
    PointProvenance, ProvenanceObservationStatus, UncertaintyStatus)
from integration.validation import ProvenanceMerger, ProvenanceResolver

def root(point_id="root", **changes):
    values=dict(point_id=point_id,source_point_ids=(),source_run_id="run",source_frame_id="frame",segment_id="seg",
      observation_status=ProvenanceObservationStatus.OBSERVED,confidence_status=ConfidenceStatus.VERIFIED,confidence_value=.9,
      uncertainty_status=UncertaintyStatus.KNOWN,timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",
      calibration_id="cal",transform_id="tf",config_version="cfg",derivation_method=DerivationMethod.OBSERVATION)
    values.update(changes);return PointProvenance(**values)

class IntegrityTests(unittest.TestCase):
    def test_key_mismatch_and_lineage_compatibility_are_enforced(self):
        item=root("actual")
        with self.assertRaisesRegex(ValueError,"key must equal"):
            ProvenanceResolver({"different":item})
        source=root("source",coordinate_frame="camera")
        derived=PointProvenance("derived",("source",),"run","frame","seg",ProvenanceObservationStatus.OBSERVED,ConfidenceStatus.VERIFIED,.8,UncertaintyStatus.KNOWN,1.,"sim","workpiece_m","m","cal","tf","cfg",DerivationMethod.MIDPOINT)
        outcome=ProvenanceResolver({"source":source,"derived":derived}).validate("derived")
        self.assertIn("COORDINATE_FRAME_MISMATCH",outcome.reason_codes)

    def test_merge_preserves_lineage_without_trust_upgrade(self):
        left,right=root("left"),root("right")
        result=ProvenanceMerger().merge("mid",(left,right),derivation_method=DerivationMethod.MIDPOINT,
            source_run_id="run",source_frame_id="frame",segment_id="seg",timestamp=1.,clock_domain="sim",
            coordinate_frame="workpiece_m",units="m",calibration_id="cal",transform_id="tf",config_version="cfg")
        self.assertEqual(result.reason_codes,())
        self.assertEqual(result.point.source_point_ids,("left","right"))
        self.assertEqual(result.point.confidence_status,ConfidenceStatus.UNKNOWN)
        self.assertEqual(result.point.uncertainty_status,UncertaintyStatus.UNKNOWN)
        self.assertIn("CONFIDENCE_UNKNOWN",result.point.strict_reason_codes())

    def test_merge_rejects_predicted_unknown_and_incompatible_sources(self):
        predicted=root("p",observation_status=ProvenanceObservationStatus.PREDICTED)
        unknown=root("u",observation_status=ProvenanceObservationStatus.UNKNOWN)
        for source,code in ((predicted,"SOURCE_PREDICTED"),(unknown,"SOURCE_OBSERVATION_UNKNOWN")):
            result=ProvenanceMerger().merge("out",(source,),derivation_method=DerivationMethod.SMOOTHING,
              source_run_id="run",source_frame_id="frame",segment_id="seg",timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",calibration_id="cal",transform_id="tf",config_version="cfg")
            self.assertIsNone(result.point);self.assertIn(code,result.reason_codes)
        mismatch=root("x",units="mm")
        result=ProvenanceMerger().merge("out",(mismatch,),derivation_method=DerivationMethod.RESAMPLING,
          source_run_id="run",source_frame_id="frame",segment_id="seg",timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",calibration_id="cal",transform_id="tf",config_version="cfg")
        self.assertIn("UNITS_MISMATCH",result.reason_codes)

    def test_contract_and_timing_invariants_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"ACCEPTED"):
            IntegrationRecord("run","corr","stage",AdmissionStatus.ACCEPTED,reason_codes=("REJECTED",))
        with self.assertRaisesRegex(ValueError,"unknown confidence"):
            root(confidence_status=ConfidenceStatus.UNKNOWN,confidence_value=.2)
        with self.assertRaisesRegex(ValueError,"malformed source_frame_id"):
            root(source_frame_id=True)
        with self.assertRaisesRegex(ValueError,"unknown uncertainty"):
            root(uncertainty_status="KNOWN")
        timing=MotionTimingRecord(0,"seg","path","workpiece_m","m",1.,None,.1,0.,AdmissionStatus.ACCEPTED)
        self.assertEqual(timing.timing_reason_codes(),("TIMING_UNSPECIFIED",))

    def test_nested_run_result_and_manual_mutation_checks(self):
        blocked=IntegrationRecord("run","corr","stage",AdmissionStatus.BLOCKED)
        event=IntegrationEvent("run","corr","stage","BLOCKED",1.)
        with self.assertRaisesRegex(ValueError,"blocked record"):
            IntegrationRunResult("run","corr",(blocked,),(event,),AdmissionStatus.REJECTED,True)
        with self.assertRaisesRegex(ValueError,"record run/correlation"):
            IntegrationRunResult("run","corr",(IntegrationRecord("other","corr","stage",AdmissionStatus.REJECTED),),(event,),AdmissionStatus.REJECTED,True)
        # Manual mutation cases: if either guard is removed, a bad contract becomes constructible.
        with self.assertRaises(ValueError):
            IntegrationRecord("run","corr","stage",AdmissionStatus.ACCEPTED,reason_codes=("MUTANT_REASON",))
        with self.assertRaises(ValueError):
            ProvenanceResolver({"wrong":root("right")})

if __name__=="__main__":unittest.main()
