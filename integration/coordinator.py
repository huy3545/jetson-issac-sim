"""Stage 7B-3 contract-only FSM coordinator; it never executes motion."""
from dataclasses import dataclass
from enum import Enum
import math
from uuid import uuid4

from .contracts import INTEGRATION_SCHEMA_VERSION
from .independent_safety import (IndependentSafetyValidator, PathPlan, MotionPlan, SafetyDecision,
    SafetyStatus, SyntheticMachineLimits, SyntheticMachineState, apply_boundary_veto)

class IntegrationState(str,Enum): IDLE="IDLE";SCAN="SCAN";PREVIEW="PREVIEW";CONFIRM="CONFIRM";CUT="CUT";HOLD="HOLD";FAULT="FAULT"
_TRANSITIONS={IntegrationState.IDLE:{IntegrationState.SCAN},IntegrationState.SCAN:{IntegrationState.PREVIEW,IntegrationState.HOLD,IntegrationState.FAULT},IntegrationState.PREVIEW:{IntegrationState.CONFIRM,IntegrationState.HOLD,IntegrationState.FAULT},IntegrationState.CONFIRM:{IntegrationState.CUT,IntegrationState.HOLD,IntegrationState.FAULT},IntegrationState.CUT:{IntegrationState.HOLD,IntegrationState.FAULT},IntegrationState.HOLD:{IntegrationState.IDLE,IntegrationState.FAULT},IntegrationState.FAULT:{IntegrationState.IDLE}}
BOUNDARY_REGISTRY={"vision-path":False,"path-motion":False,"motion-safety":False,"safety-simulation-cut":True}

@dataclass(frozen=True)
class SynchronizationPolicy:
    version:str; max_delta_s:float
    def __post_init__(self):
        if not self.version or isinstance(self.max_delta_s,bool) or not isinstance(self.max_delta_s,(int,float)) or not math.isfinite(self.max_delta_s) or self.max_delta_s<0:raise ValueError("invalid synchronization policy")
@dataclass(frozen=True)
class TransitionResult:
    previous:IntegrationState;next:IntegrationState;status:SafetyStatus;reason_codes:tuple[str,...]=()
@dataclass(frozen=True)
class CoordinatorAuditEvent:
    event_id:str;run_id:str;correlation_id:str;stage_id:str;boundary_id:str|None;previous_state:IntegrationState;next_state:IntegrationState;input_identity:str;path_hash:str|None;motion_hash:str|None;safety_reference:str|None;status:SafetyStatus;reason_codes:tuple[str,...];timestamp:float;clock_domain:str;config_version:str;audit_complete:bool;schema_version:str=INTEGRATION_SCHEMA_VERSION
@dataclass(frozen=True)
class IntegrationCoordinatorResult:
    run_id:str;correlation_id:str;final_state:IntegrationState;status:SafetyStatus;safety_decision:SafetyDecision|None;boundaries:tuple;transitions:tuple[TransitionResult,...];audit_events:tuple[CoordinatorAuditEvent,...];audit_complete:bool;execution_authorized:bool=False;schema_version:str=INTEGRATION_SCHEMA_VERSION
    def __post_init__(self):
        if self.execution_authorized or self.schema_version!=INTEGRATION_SCHEMA_VERSION:raise ValueError("invalid coordinator result")
        if not self.audit_complete and self.status is SafetyStatus.ACCEPTED:raise ValueError("incomplete audit cannot be accepted")

class IntegrationCoordinator:
    def __init__(self, validator=None):self.validator=validator or IndependentSafetyValidator();self.state=IntegrationState.IDLE;self.last_reset_audit=None
    def transition(self,next_state,reasons=()):
        previous=self.state
        if next_state not in _TRANSITIONS[previous]:return TransitionResult(previous,previous,SafetyStatus.REJECTED,("ILLEGAL_FSM_TRANSITION",))
        self.state=next_state;return TransitionResult(previous,next_state,SafetyStatus.ACCEPTED,tuple(reasons))
    def _sync(self,path,motion,state,policy):
        if not isinstance(state,SyntheticMachineState):return ("MACHINE_STATE_MISSING",)
        if state.clock_domain!=path.clock_domain or motion.clock_domain!=path.clock_domain:return ("CLOCK_DOMAIN_MISMATCH",)
        if (path.timestamp,motion.timestamp,state.timestamp)==(path.timestamp,)*3:return ()
        if policy is None:return ("MACHINE_STATE_TIME_UNVERIFIED",)
        if max(abs(path.timestamp-motion.timestamp),abs(path.timestamp-state.timestamp))>policy.max_delta_s:return ("MACHINE_STATE_TIME_UNVERIFIED",)
        return ()
    def boundary(self,boundary_id,upstream,safety=None,*,path=None,motion=None,limits=None):
        if boundary_id not in BOUNDARY_REGISTRY:return None,("UNKNOWN_BOUNDARY_ID",)
        return apply_boundary_veto(boundary_id,boundary_id,upstream,safety,protected=BOUNDARY_REGISTRY[boundary_id],path=path,motion=motion,limits=limits),()
    def coordinate(self,path,motion,limits,state,*,sync_policy=None,audit_sink=None):
        run=getattr(path,"run_id","unknown");corr=getattr(path,"correlation_id","unknown");events=[];trans=[];boundaries=[];complete=True
        def event(stage,boundary,result,reasons=()):
            nonlocal complete
            previous=getattr(result,"previous",self.state);next_state=getattr(result,"next",self.state)
            item=CoordinatorAuditEvent(str(len(events)),run,corr,stage,boundary,previous,next_state,boundary or stage,getattr(path,"content_hash",None),getattr(motion,"content_hash",None),getattr(result,"motion_plan_hash",None),result.status if result else SafetyStatus.BLOCKED,tuple(reasons),getattr(path,"timestamp",0.),getattr(path,"clock_domain","unknown"),getattr(path,"config_version","unknown"),complete)
            events.append(item)
            if audit_sink:
                try:audit_sink.emit(item);return True
                except Exception:complete=False;return False
            return True
        def hold(reasons,stage):
            item=self.transition(IntegrationState.HOLD,tuple(dict.fromkeys(tuple(reasons)+("EVIDENCE_INCOMPLETE",))));trans.append(item);event(stage,None,item,item.reason_codes);return self._result(run,corr,decision if 'decision' in locals() else None,boundaries,trans,events,complete)
        start=self.transition(IntegrationState.SCAN);trans.append(start);event("fsm",None,start,start.reason_codes)
        if not complete:return hold(("EVIDENCE_INCOMPLETE",),"audit_failure")
        if start.status is not SafetyStatus.ACCEPTED:return self._result(run,corr,None,boundaries,trans,events,complete)
        sync=self._sync(path,motion,state,sync_policy)
        if sync:
            hold=self.transition(IntegrationState.HOLD,sync);trans.append(hold);event("sync",None,hold,sync);return self._result(run,corr,None,boundaries,trans,events,complete)
        preview=self.transition(IntegrationState.PREVIEW);trans.append(preview);event("preview",None,preview)
        if not complete:return hold(("EVIDENCE_INCOMPLETE",),"audit_failure")
        decision,audit_ok=self.validator.assess(path,motion,limits,state);complete=complete and audit_ok;event("safety","motion-safety",decision,decision.reason_codes)
        veto,error=self.boundary("safety-simulation-cut",SafetyStatus.ACCEPTED,decision,path=path,motion=motion,limits=limits);boundaries.append(veto)
        if error or veto.status is not SafetyStatus.ACCEPTED or not complete:
            hold=self.transition(IntegrationState.HOLD,tuple(error)+veto.reason_codes+(() if complete else ("EVIDENCE_INCOMPLETE",)));trans.append(hold);event("boundary","safety-simulation-cut",hold,hold.reason_codes);return self._result(run,corr,decision,boundaries,trans,events,complete)
        confirm=self.transition(IntegrationState.CONFIRM);trans.append(confirm);event("confirm",None,confirm)
        if not complete:return hold(("EVIDENCE_INCOMPLETE",),"audit_failure")
        cut=self.transition(IntegrationState.CUT);trans.append(cut);event("cut","safety-simulation-cut",cut)
        if not complete:return hold(("EVIDENCE_INCOMPLETE",),"audit_failure")
        return self._result(run,corr,decision,boundaries,trans,events,complete)
    def reset(self,*,audit_sink=None,run_id="reset",correlation_id="reset",timestamp=0.,clock_domain="synthetic",config_version="synthetic"):
        previous=self.state
        candidate=TransitionResult(previous,IntegrationState.IDLE,SafetyStatus.ACCEPTED,("RESET",)) if IntegrationState.IDLE in _TRANSITIONS[previous] else TransitionResult(previous,previous,SafetyStatus.REJECTED,("ILLEGAL_FSM_TRANSITION",))
        audit=CoordinatorAuditEvent("reset",run_id,correlation_id,"reset",None,candidate.previous,candidate.next,"reset",None,None,None,candidate.status,candidate.reason_codes,timestamp,clock_domain,config_version,True)
        self.last_reset_audit=audit
        try:
            if audit_sink:audit_sink.emit(audit)
        except Exception:
            self.last_reset_audit=CoordinatorAuditEvent("reset",run_id,correlation_id,"reset",None,previous,previous,"reset",None,None,None,SafetyStatus.BLOCKED,("EVIDENCE_INCOMPLETE",),timestamp,clock_domain,config_version,False)
            return TransitionResult(previous,previous,SafetyStatus.BLOCKED,("EVIDENCE_INCOMPLETE",))
        if candidate.status is SafetyStatus.ACCEPTED:self.state=IntegrationState.IDLE
        return candidate
    def _result(self,run,corr,decision,boundaries,trans,events,complete):
        status=SafetyStatus.ACCEPTED if self.state is IntegrationState.CUT and complete else SafetyStatus.BLOCKED
        return IntegrationCoordinatorResult(run,corr,self.state,status,decision,tuple(boundaries),tuple(trans),tuple(events),complete)
