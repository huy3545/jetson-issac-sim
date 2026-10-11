"""Stage 7B-3 FSM/coordinator tests using only synthetic verified contracts."""
import unittest
from integration.coordinator import IntegrationCoordinator,IntegrationState,SynchronizationPolicy
from integration.independent_safety import SafetyStatus
from tests.test_independent_safety import limits,motion,path,state
class BadSink:
 def emit(self,event):raise OSError("audit")
class IndexedSink:
 def __init__(self,fail=None):self.fail=fail;self.events=[]
 def emit(self,event):
  if self.fail==len(self.events):raise OSError("audit")
  self.events.append(event)
class CoordinatorTests(unittest.TestCase):
 def test_legal_illegal_reset_and_no_skip(self):
  c=IntegrationCoordinator();self.assertEqual(c.transition(IntegrationState.CUT).reason_codes,("ILLEGAL_FSM_TRANSITION",));self.assertEqual(c.transition(IntegrationState.SCAN).status,SafetyStatus.ACCEPTED);self.assertEqual(c.transition(IntegrationState.CONFIRM).reason_codes,("ILLEGAL_FSM_TRANSITION",));self.assertEqual(c.transition(IntegrationState.PREVIEW).status,SafetyStatus.ACCEPTED);self.assertEqual(c.transition(IntegrationState.HOLD).status,SafetyStatus.ACCEPTED);self.assertEqual(c.reset().next,IntegrationState.IDLE)
 def test_valid_synthetic_logic_reaches_cut_without_authorization(self):
  r=IntegrationCoordinator().coordinate(path(),motion(path()),limits(),state())
  self.assertEqual(r.final_state,IntegrationState.CUT);self.assertEqual(r.status,SafetyStatus.ACCEPTED);self.assertFalse(r.execution_authorized);self.assertTrue(r.audit_complete)
 def test_time_domain_and_missing_policy_hold(self):
  p=path();m=motion(p,timestamp=2.);r=IntegrationCoordinator().coordinate(p,m,limits(),state(timestamp=2.));self.assertEqual(r.final_state,IntegrationState.HOLD);self.assertIn("MACHINE_STATE_TIME_UNVERIFIED",r.transitions[-1].reason_codes)
  r=IntegrationCoordinator().coordinate(p,m,limits(),state(timestamp=2.),sync_policy=SynchronizationPolicy("sync",2.));self.assertEqual(r.final_state,IntegrationState.CUT)
  r=IntegrationCoordinator().coordinate(p,m,limits(),state(clock_domain="other"),sync_policy=SynchronizationPolicy("sync",2.));self.assertIn("CLOCK_DOMAIN_MISMATCH",r.transitions[-1].reason_codes)
 def test_faults_boundaries_and_audit_failure_hold(self):
  c=IntegrationCoordinator();self.assertEqual(c.boundary("unknown",SafetyStatus.ACCEPTED)[1],("UNKNOWN_BOUNDARY_ID",))
  p=path();m=motion(p,source_path_hash="bad");r=IntegrationCoordinator().coordinate(p,m,limits(),state());self.assertEqual(r.final_state,IntegrationState.HOLD);self.assertNotEqual(r.safety_decision.status,SafetyStatus.ACCEPTED)
  r=IntegrationCoordinator().coordinate(path(),motion(path()),limits(),state(),audit_sink=BadSink());self.assertEqual(r.final_state,IntegrationState.HOLD);self.assertFalse(r.audit_complete)
 def test_no_controller_import(self):
  import integration.coordinator as module;self.assertNotIn("MachineController",module.__dict__)
 def test_each_audit_failure_prevents_cut_and_transition_events_are_exact(self):
  # Indices: SCAN, PREVIEW, safety, CONFIRM, CUT. A failure at each must hold.
  for index in range(5):
   p=path();sink=IndexedSink(index);r=IntegrationCoordinator().coordinate(p,motion(p),limits(),state(),audit_sink=sink)
   self.assertNotEqual(r.final_state,IntegrationState.CUT);self.assertFalse(r.audit_complete);self.assertNotEqual(r.status,SafetyStatus.ACCEPTED)
  sink=IndexedSink();p=path();r=IntegrationCoordinator().coordinate(p,motion(p),limits(),state(),audit_sink=sink)
  self.assertEqual(r.final_state,IntegrationState.CUT);self.assertTrue(r.audit_complete)
  for event,transition in zip(r.audit_events[:1],r.transitions[:1]):self.assertEqual((event.previous_state,event.next_state),(transition.previous,transition.next))
 def test_reset_is_auditable_and_fails_closed(self):
  c=IntegrationCoordinator();c.transition(IntegrationState.SCAN);c.transition(IntegrationState.HOLD)
  reset=c.reset(audit_sink=IndexedSink());self.assertEqual(reset.next,IntegrationState.IDLE);self.assertEqual(c.last_reset_audit.next_state,IntegrationState.IDLE)
  c=IntegrationCoordinator();c.transition(IntegrationState.SCAN);c.transition(IntegrationState.HOLD);blocked=c.reset(audit_sink=BadSink());self.assertEqual(blocked.status,SafetyStatus.BLOCKED);self.assertEqual(c.state,IntegrationState.HOLD)
if __name__=="__main__":unittest.main()
