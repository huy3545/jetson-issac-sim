"""Experimental corner-aware extension around the public Phase 1A solver."""
from dataclasses import dataclass
import numpy as np
from .planar_medial_axis import (PlanarMedialAxisSolver, SolverDiagnostics,
                                 SolverResult, SolverSegment, SolverStatus)

@dataclass(frozen=True)
class CornerAwareConfig:
    corner_angle_deg:float=30.0
    corner_fraction_tolerance:float=.15
    resample_step_m:float=.005
    experimental_touch_tol_m:float=1e-6

def _turns(points,cfg):
    points=np.asarray(points,float); result=[]
    for index in range(1,len(points)-1):
        before=points[index]-points[index-1]; after=points[index+1]-points[index]
        nb,na=np.linalg.norm(before),np.linalg.norm(after)
        if nb<=1e-12 or na<=1e-12:return None,"ZERO_LENGTH_SEGMENT"
        dot=np.clip((before@after)/(nb*na),-1.,1.); angle=float(np.degrees(np.arccos(dot)))
        if angle>=cfg.corner_angle_deg:
            cross=before[0]*after[1]-before[1]*after[0]
            result.append((index,angle,int(np.sign(cross))))
    return result,None
def _length(points): return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
def _resample(points,count):
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    target=np.linspace(0.,distance[-1],count)
    return np.column_stack([np.interp(target,distance,points[:,axis]) for axis in range(2)])

class CornerAwareTwoLineSolver:
    """Preserves one matched 90° source corner; no reflex/turnback inference."""
    def __init__(self,config=CornerAwareConfig(),base=None): self.config=config;self.base=base or PlanarMedialAxisSolver()
    def solve(self,left,right):
        left,right=np.asarray(left,float),np.asarray(right,float)
        # Reuse Phase 1A for its topology validation and every smooth case.
        baseline=self.base.solve(left,right)
        turns_left,error_left=_turns(left,self.config); turns_right,error_right=_turns(right,self.config)
        if error_left or error_right:return baseline
        if not turns_left and not turns_right:return baseline
        if len(turns_left)!=1 or len(turns_right)!=1:
            return self._reject("MULTI_CORNER_UNSUPPORTED")
        (li,la,ls),(ri,ra,rs)=turns_left[0],turns_right[0]
        if la>=180.-1e-9 or ra>=180.-1e-9:return self._reject("TURNBACK_UNSUPPORTED")
        # Opposite signed turns require a declared polygon orientation to tell
        # convex from reflex; this open-boundary contract intentionally lacks it.
        if ls==0 or rs==0 or ls!=rs:return self._reject("REFLEX_ORIENTATION_UNPROVEN")
        lf,rf=_length(left[:li+1])/_length(left),_length(right[:ri+1])/_length(right)
        if abs(lf-rf)>self.config.corner_fraction_tolerance:return self._reject("AMBIGUOUS_CORNER_CORRESPONDENCE")
        # Existing Phase 1A remains the topology gate: never construct a path
        # when it already reports invalid input/boundary contact/intersection.
        if baseline.overall_status is not SolverStatus.CONVERGED:return baseline
        runs=[]
        for lrun,rrun in ((left[:li+1],right[:ri+1]),(left[li:],right[ri:])):
            count=max(2,int(np.ceil(max(_length(lrun),_length(rrun))/self.config.resample_step_m))+1)
            run=(_resample(lrun,count)+_resample(rrun,count))/2
            runs.append(run)
        xy=np.vstack((runs[0],runs[1][1:]))
        ds=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        if not np.all(ds>0):return self._reject("CENTERLINE_DEGENERATE")
        s=np.r_[0.,np.cumsum(ds)]; widths=np.empty(len(xy)); widths.fill(np.nan)
        segment=SolverSegment(xy,s,widths,np.ones(len(xy),bool),SolverStatus.CONVERGED,("CORNER_AWARE_ARC_LENGTH_CORRESPONDENCE",))
        diagnostics=SolverDiagnostics(coverage_ratio=1.,excluded=("CORNER_AWARE_ARC_LENGTH_CORRESPONDENCE",))
        return SolverResult((segment,),SolverStatus.CONVERGED,True,diagnostics)
    def _reject(self,code):
        return SolverResult((),SolverStatus.AMBIGUOUS,False,SolverDiagnostics(excluded=(code,)))
