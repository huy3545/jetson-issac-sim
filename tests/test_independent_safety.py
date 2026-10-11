"""Stage 7B-2 deterministic safety-contract tests; no controller imports."""
import unittest
import numpy as np
from integration.contracts import (AdmissionStatus,ConfidenceStatus,DerivationMethod,MotionTimingRecord,PointProvenance,ProvenanceObservationStatus,UncertaintyStatus)
from integration.independent_safety import IndependentSafetyValidator,MachineStateEvidenceStatus,MotionPlan,PathPlan,SafetyStatus,SyntheticMachineLimits,SyntheticMachineState,apply_boundary_veto
def point(i):return PointProvenance(f"p{i}",(),"run","frame","seg",ProvenanceObservationStatus.OBSERVED,ConfidenceStatus.VERIFIED,.9,UncertaintyStatus.KNOWN,1.,"sim","workpiece_m","m","cal","tf","cfg",DerivationMethod.OBSERVATION)
def path(**changes):
 pts=(point(0),point(1));d=dict(run_id="run",correlation_id="corr",path_id="path",source_frame_id="frame",timestamp=1.,clock_domain="sim",coordinate_frame="workpiece_m",units="m",calibration_id="cal",transform_id="tf",config_version="cfg",segment_ids=("seg",),point_ids=("p0","p1"),point_provenance=pts,coordinates=np.array([[0.,0.,.5],[.01,0.,.5]]),geometry_status="VALID",qualification_status="VERIFIED",uncertainty_status=UncertaintyStatus.KNOWN,has_gap=False);d.update(changes);return PathPlan(**d)
def motion(p,**changes):
 timing=tuple(MotionTimingRecord(i,"seg",p.path_id,"workpiece_m","m",1.,.1,.01,0.,AdmissionStatus.ACCEPTED) for i in range(2));d=dict(run_id="run",correlation_id="corr",motion_plan_id="motion",source_path_id=p.path_id,source_path_hash=p.content_hash,segment_ids=("seg",),target_ids=("t0","t1"),target_point_ids=p.point_ids,target_coordinates=p.coordinates,coordinate_frame="workpiece_m",units="m",timestamp=1.,clock_domain="sim",config_version="cfg",timing_records=timing,provenance_references=("path",),motion_status="VALID");d.update(changes);return MotionPlan(**d)
def limits():return SyntheticMachineLimits("limits",(-1.,-1.,0.),(1.,1.,1.),.1,.1)
def state(**changes):
 d=dict(run_id="run",correlation_id="corr",source_frame_id="frame",timestamp=1.,clock_domain="sim",configuration_id="limits",values=(("z",.5),),evidence_status=MachineStateEvidenceStatus.KNOWN);d.update(changes);return SyntheticMachineState(**d)
class Sink:
 def emit(self,event):raise OSError("sink")
class SafetyTests(unittest.TestCase):
 def test_immutable_plans_and_canonical_hash(self):
  raw=np.array([[0.,0.,.5],[.01,0.,.5]]);p=path(coordinates=raw);raw[0,0]=9;self.assertEqual(p.coordinates[0,0],0.)
  with self.assertRaises(ValueError):p.coordinates[0,0]=9
  self.assertEqual(p.content_hash,path().content_hash);self.assertNotEqual(p.content_hash,path(coordinates=np.array([[0.,0.,.5],[.02,0.,.5]])).content_hash);self.assertNotEqual(p.content_hash,path(calibration_id="other").content_hash)
 def test_motion_immutable_and_hash_reference(self):
  p=path();m=motion(p);self.assertNotEqual(m.content_hash,motion(p,motion_status="OTHER").content_hash)
  with self.assertRaises(ValueError):m.target_coordinates[0,0]=3
 def test_valid_plan_is_simulation_accepted_but_not_authorized(self):
  p=path();d,complete=IndependentSafetyValidator().assess(p,motion(p),limits(),machine_state=state());self.assertEqual(d.status,SafetyStatus.ACCEPTED);self.assertFalse(d.execution_authorized);self.assertTrue(complete)
 def test_hash_provenance_gap_unknown_and_machine_missing_fail_closed(self):
  p=path();v=IndependentSafetyValidator();self.assertIn("PATH_HASH_MISMATCH",v.assess(p,motion(p,source_path_hash="wrong"),limits(),state())[0].reason_codes);self.assertIn("MACHINE_STATE_MISSING",v.assess(p,motion(p),limits(),None)[0].reason_codes)
  gap=path(has_gap=True);self.assertIn("PATH_GAP",v.assess(gap,motion(gap),limits(),state())[0].reason_codes);unknown=path(uncertainty_status=UncertaintyStatus.UNKNOWN);self.assertIn("UNCERTAINTY_UNKNOWN",v.assess(unknown,motion(unknown),limits(),state())[0].reason_codes)
  bad_point=PointProvenance("p0",(),"run","frame","seg",ProvenanceObservationStatus.PREDICTED,ConfidenceStatus.VERIFIED,.9,UncertaintyStatus.KNOWN,1.,"sim","workpiece_m","m","cal","tf","cfg",DerivationMethod.OBSERVATION)
  bad_path=path(point_provenance=(bad_point,point(1)));self.assertIn("POINT_NOT_OBSERVED",v.assess(bad_path,motion(bad_path),limits(),state())[0].reason_codes)
 def test_limits_timing_segment_and_boundary_veto(self):
  p=path();v=IndependentSafetyValidator();far=motion(p,target_coordinates=np.array([[0.,0.,.5],[.5,0.,.5]]));self.assertIn("TARGET_STEP_EXCEEDED",v.assess(p,far,limits(),state())[0].reason_codes)
  timing=tuple(MotionTimingRecord(i,"seg",p.path_id,"workpiece_m","m",1.,None,.01,0.,AdmissionStatus.ACCEPTED) for i in range(2));self.assertIn("TIMING_UNSPECIFIED",v.assess(p,motion(p,timing_records=timing),limits(),state())[0].reason_codes)
  d,_=v.assess(p,motion(p,segment_ids=("other",)),limits(),state());self.assertIn("SEGMENT_ORDER_MISMATCH",d.reason_codes);self.assertEqual(apply_boundary_veto("motion-safety","motion",SafetyStatus.ACCEPTED,d).status,SafetyStatus.BLOCKED)
 def test_audit_failure_and_no_controller(self):
  p=path();d,complete=IndependentSafetyValidator().assess(p,motion(p),limits(),state(),Sink());self.assertFalse(complete);self.assertIn("EVIDENCE_INCOMPLETE",d.reason_codes);self.assertEqual(d.status,SafetyStatus.BLOCKED)
  import integration.independent_safety as module;self.assertNotIn("MachineController",module.__dict__)
if __name__=="__main__":unittest.main()
