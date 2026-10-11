"""Stage 7B-2A protected-boundary and immutable-buffer regression tests."""
import unittest
import numpy as np
from integration.contracts import AdmissionStatus, MotionTimingRecord
from integration.independent_safety import (IndependentSafetyValidator,MachineStateEvidenceStatus,SafetyStatus,SyntheticMachineState,apply_boundary_veto)
from tests.test_independent_safety import limits,motion,path,state

class BoundaryHardeningTests(unittest.TestCase):
 def test_protected_boundary_requires_current_matching_decision(self):
  p=path();m=motion(p);d,_=IndependentSafetyValidator().assess(p,m,limits(),state())
  self.assertEqual(apply_boundary_veto("safety-mock","m",SafetyStatus.ACCEPTED,protected=True,path=p,motion=m,limits=limits()).reason_codes,("SAFETY_DECISION_MISSING",))
  stale=path(timestamp=2.);self.assertIn("SAFETY_DECISION_IDENTITY_MISMATCH",apply_boundary_veto("safety-mock","m",SafetyStatus.ACCEPTED,d,protected=True,path=stale,motion=motion(stale),limits=limits()).reason_codes)
  unrelated=motion(p,motion_plan_id="other");self.assertIn("SAFETY_DECISION_IDENTITY_MISMATCH",apply_boundary_veto("safety-mock","m",SafetyStatus.ACCEPTED,d,protected=True,path=p,motion=unrelated,limits=limits()).reason_codes)
  self.assertEqual(apply_boundary_veto("vision-path","v",SafetyStatus.ACCEPTED).status,SafetyStatus.ACCEPTED)
 def test_machine_state_and_contract_consistency_fail_closed(self):
  p=path();m=motion(p);v=IndependentSafetyValidator()
  for changed,code in ((dict(clock_domain="other"),"MACHINE_STATE_CLOCK_DOMAIN_MISMATCH"),(dict(configuration_id="bad"),"MACHINE_STATE_CONFIG_MISMATCH"),(dict(evidence_status=MachineStateEvidenceStatus.UNKNOWN),"MACHINE_STATE_EVIDENCE_UNKNOWN")):
   self.assertIn(code,v.assess(p,m,limits(),state(**changed))[0].reason_codes)
  wrong_point=path(point_provenance=(p.point_provenance[0],p.point_provenance[1].__class__("p1",(),"run","other","seg",p.point_provenance[1].observation_status,p.point_provenance[1].confidence_status,.9,p.point_provenance[1].uncertainty_status,1.,"sim","workpiece_m","m","cal","tf","cfg",p.point_provenance[1].derivation_method)))
  self.assertIn("POINT_PATH_METADATA_MISMATCH",v.assess(wrong_point,motion(wrong_point),limits(),state())[0].reason_codes)
  timings=(MotionTimingRecord(1,"seg",p.path_id,"workpiece_m","m",1.,.1,.01,0.,AdmissionStatus.ACCEPTED),MotionTimingRecord(0,"bad", "other","workpiece_m","m",1.,.1,.01,0.,AdmissionStatus.ACCEPTED))
  reasons=v.assess(p,motion(p,timing_records=timings),limits(),state())[0].reason_codes
  for code in ("TIMING_TARGET_INDEX_MISMATCH","TIMING_SOURCE_PATH_MISMATCH","TIMING_SEGMENT_MISMATCH"):self.assertIn(code,reasons)
 def test_byte_backed_arrays_reject_alias_view_and_write_flag_mutations(self):
  raw=np.array([[0.,0.,.5],[.01,0.,.5]]);p=path(coordinates=raw);m=motion(p);raw[:]=9
  for array in (p.coordinates,m.target_coordinates,p.coordinates.view(),m.target_coordinates.view()):
   with self.assertRaises(ValueError):array.setflags(write=True)
   self.assertFalse(array.flags.writeable)
  self.assertEqual(p.coordinates[0,0],0.);self.assertEqual(m.target_coordinates[0,0],0.)
if __name__=="__main__":unittest.main()
