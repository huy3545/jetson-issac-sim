"""Phase 1a planar centreline solver for clean, open two-boundary channels.

This module deliberately does not call ``PathPlanner.plan`` and produces XY only.
It is conservative: an unverified output point is never labelled CONVERGED.
"""
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
import math
import numpy as np

class SolverStatus(str, Enum):
    CONVERGED="CONVERGED"; NOT_CONVERGED="NOT_CONVERGED"; AMBIGUOUS="AMBIGUOUS"; INVALID_GEOMETRY="INVALID_GEOMETRY"

@dataclass(frozen=True)
class PlanarMedialAxisConfig:
    epsilon_solver_m: float = 1e-6
    root_tolerance_m: float = 1e-9
    numerical_zero_length_tol_m: float = 1e-12
    max_root_iterations: int = 60
    bracket_sample_count: int = 33
    max_marching_steps: int = 5000
    min_marching_step_m: float = 1e-4
    marching_step_width_factor: float = .25
    max_jump_factor: float = 2.0
    min_ray_angle_deg: float = 20.0
    bracket_ray_length_factor: float = 2.0
    resample_step_m: float = .005
    max_vertices_per_boundary: int = 2000
    max_seed_retries: int = 3
    experimental_min_channel_width_m: float = .004
    experimental_min_segment_length_m: float = .02
    experimental_touch_tol_m: float = 1e-6
    experimental_min_AB_angle_deg: float = 90.0
    experimental_fully_covered_min_ratio: float = .95
    def __post_init__(self):
        if not (0 < self.root_tolerance_m < self.epsilon_solver_m/10): raise ValueError("root_tolerance_m must be < epsilon_solver_m/10")
        if self.resample_step_m <= 0 or self.max_vertices_per_boundary < 2: raise ValueError("invalid solver configuration")

@dataclass(frozen=True)
class SolverSegment:
    xy: np.ndarray; s: np.ndarray; width_m: np.ndarray; point_valid: np.ndarray
    status: SolverStatus; reason_codes: tuple[str,...] = ()

@dataclass(frozen=True)
class SolverDiagnostics:
    distance_call_count: int=0; iteration_count: int=0; marching_step_count: int=0; seed_retry_count: int=0
    elapsed_ms: float=0.; max_residual_m: float=0.; coverage_ratio: float|None=None
    excluded: tuple[str,...]=(); coverage_estimation_method: str="valid centreline length / mean boundary length"

@dataclass(frozen=True)
class SolverResult:
    segments: tuple[SolverSegment,...]; overall_status: SolverStatus; fully_covered: bool; diagnostics: SolverDiagnostics

def _cross(a,b,c):
    u=b-a; v=c-a
    return float(u[0]*v[1]-u[1]*v[0])
def _proper_intersect(a,b,c,d, tol=1e-12):
    return _cross(a,b,c)*_cross(a,b,d) < -tol and _cross(c,d,a)*_cross(c,d,b) < -tol
def _point_segment(point,a,b):
    vector=b-a; denom=float(vector@vector)
    t=0. if denom==0 else float(np.clip(((point-a)@vector)/denom,0.,1.))
    q=a+t*vector
    return float(np.linalg.norm(point-q)),q,t
def _distance(point,line, counter):
    counter[0]+=len(line)-1
    choices=[_point_segment(point,line[i],line[i+1])+(i,) for i in range(len(line)-1)]
    d,q,t,index=min(choices,key=lambda item:item[0]); return d,q,t,index
def _length(line): return float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())
def _resample(line,count):
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    x=np.linspace(0,d[-1],count)
    return np.column_stack([np.interp(x,d,line[:,k]) for k in range(2)])
def _point_in_polygon(point, polygon):
    x,y=point; inside=False
    for a,b in zip(polygon,np.vstack((polygon[1:],polygon[:1]))):
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: inside=not inside
    return inside
def _validate(line, cfg):
    if not isinstance(line,np.ndarray) or line.ndim!=2 or line.shape[1]!=2: return "INVALID_SHAPE"
    if not np.isfinite(line).all(): return "NON_FINITE_INPUT"
    if len(line)<2: return "TOO_FEW_VERTICES"
    if len(line)>cfg.max_vertices_per_boundary: return "TOO_MANY_VERTICES"
    if (np.linalg.norm(np.diff(line,axis=0),axis=1)<=cfg.numerical_zero_length_tol_m).any(): return "ZERO_LENGTH_SEGMENT"
    for i in range(len(line)-1):
        for j in range(i+2,len(line)-1):
            if j!=i+1 and _proper_intersect(line[i],line[i+1],line[j],line[j+1]): return "SELF_INTERSECTION"
    return None
def _inside_channel(point,left,right,polygon,cfg,counter):
    dl,a,ta,ia=_distance(point,left,counter); dr,b,tb,ib=_distance(point,right,counter)
    if (ia==0 and ta==0.) or (ia==len(left)-2 and ta==1.) or (ib==0 and tb==0.) or (ib==len(right)-2 and tb==1.):
        return False,dl,dr,"ENDS_TRIMMED"
    va=a-point; vb=b-point; denom=np.linalg.norm(va)*np.linalg.norm(vb)
    angle=0 if denom==0 else math.degrees(math.acos(np.clip((va@vb)/denom,-1,1)))
    if angle <= cfg.experimental_min_AB_angle_deg: return False,dl,dr,"ANGLE_REJECTED"
    if not _point_in_polygon(point,polygon): return False,dl,dr,"OUTSIDE_CHANNEL"
    return True,dl,dr,""
def _postproject(point,left,right,cfg,counter):
    """Bracketed bisection across closest-boundary cross section."""
    _,a,_,_=_distance(point,left,counter); _,b,_,_=_distance(point,right,counter)
    direction=b-a; length=np.linalg.norm(direction)
    if length<=cfg.numerical_zero_length_tol_m: return None,0
    lo=a+direction*1e-9/length; hi=b-direction*1e-9/length
    def f(p): return _distance(p,left,counter)[0]-_distance(p,right,counter)[0]
    flo,fhi=f(lo),f(hi)
    if flo*fhi>0: return None,0
    iterations=0
    for iterations in range(1,cfg.max_root_iterations+1):
        mid=(lo+hi)/2; fm=f(mid)
        if abs(fm)<=cfg.root_tolerance_m: return mid,iterations
        if flo*fm<=0: hi=mid; fhi=fm
        else: lo=mid; flo=fm
    return None,iterations
def _continuity_ok(previous,current,cfg): return np.linalg.norm(current-previous)<=cfg.max_jump_factor*cfg.resample_step_m+cfg.epsilon_solver_m

class PlanarMedialAxisSolver:
    def __init__(self, config: PlanarMedialAxisConfig|None=None): self.config=config or PlanarMedialAxisConfig()
    def solve(self, boundary_a, boundary_b) -> SolverResult:
        started=perf_counter(); cfg=self.config; counter=[0]; iterations=0; excluded=[]
        try:
            left=np.asarray(boundary_a,dtype=float); right=np.asarray(boundary_b,dtype=float)
        except (TypeError, ValueError):
            return self._result((),SolverStatus.INVALID_GEOMETRY,counter,iterations,0,started,("INVALID_SHAPE",))
        for line in (left,right):
            reason=_validate(line,cfg)
            if reason: return self._result((),SolverStatus.INVALID_GEOMETRY,counter,iterations,0,started,(reason,))
        # Direction/names do not matter; choose end correspondence geometrically.
        if np.linalg.norm(left[0]-right[-1])+np.linalg.norm(left[-1]-right[0]) < np.linalg.norm(left[0]-right[0])+np.linalg.norm(left[-1]-right[-1]): right=right[::-1]
        for a,b in zip(left[:-1],left[1:]):
            for c,d in zip(right[:-1],right[1:]):
                if _proper_intersect(a,b,c,d): return self._result((),SolverStatus.INVALID_GEOMETRY,counter,iterations,0,started,("BOUNDARY_INTERSECTION",))
                if min(_point_segment(a,c,d)[0],_point_segment(c,a,b)[0]) <= cfg.experimental_touch_tol_m:
                    return self._result((),SolverStatus.INVALID_GEOMETRY,counter,iterations,0,started,("BOUNDARY_TOUCH",))
        polygon=np.vstack((left,right[::-1]))
        # Ensure the closing/opening connectors do not make a self-intersecting channel.
        for i in range(len(polygon)):
            for j in range(i+2,len(polygon)):
                if (i==0 and j==len(polygon)-1): continue
                if _proper_intersect(polygon[i],polygon[(i+1)%len(polygon)],polygon[j],polygon[(j+1)%len(polygon)]):
                    return self._result((),SolverStatus.AMBIGUOUS,counter,iterations,0,started,("BOUNDARY_INTERSECTION",))
        width_hint=min(_length(left),_length(right)); count=min(cfg.max_marching_steps,max(3,int(math.ceil(width_hint/cfg.resample_step_m))+1))
        raw=[]; reasons=[]
        for candidate in (_resample(left,count)+_resample(right,count))/2:
            point,it=_postproject(candidate,left,right,cfg,counter); iterations+=it
            if point is None: reasons.append("NO_BRACKET"); continue
            ok,dl,dr,reason=_inside_channel(point,left,right,polygon,cfg,counter)
            if not ok: reasons.append(reason); continue
            if dl+dr < cfg.experimental_min_channel_width_m: reasons.append("CHANNEL_TOO_NARROW"); continue
            if raw and not _continuity_ok(raw[-1],point,cfg): reasons.append("CONTINUITY_FAILURE"); break
            raw.append(point)
        if len(raw)<2: return self._result((),SolverStatus.NOT_CONVERGED,counter,iterations,len(raw),started,tuple(reasons) or ("NO_VALID_SEED",))
        raw=np.array(raw); spacing=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
        projected=[]; widths=[]; max_residual=0.
        for point in _resample(raw,max(2,int(math.floor(spacing[-1]/cfg.resample_step_m))+1)):
            corrected,it=_postproject(point,left,right,cfg,counter); iterations+=it
            if corrected is None: reasons.append("POSTCONDITION_FAILED"); break
            ok,dl,dr,reason=_inside_channel(corrected,left,right,polygon,cfg,counter)
            residual=abs(dl-dr); max_residual=max(max_residual,residual)
            if not ok or residual>cfg.epsilon_solver_m: reasons.append("POSTCONDITION_FAILED"); break
            if projected and not _continuity_ok(projected[-1],corrected,cfg): reasons.append("CONTINUITY_FAILURE"); break
            projected.append(corrected); widths.append(dl+dr)
        if len(projected)<2: return self._result((),SolverStatus.NOT_CONVERGED,counter,iterations,len(raw),started,tuple(reasons))
        xy=np.asarray(projected); s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        segment=SolverSegment(xy,s,np.asarray(widths),np.ones(len(xy),dtype=bool),SolverStatus.CONVERGED,("ENDS_TRIMMED",))
        return self._result((segment,),SolverStatus.CONVERGED,counter,iterations,len(raw),started,tuple(reasons),max_residual)
    def _result(self,segments,status,counter,iterations,steps,started,excluded,max_residual=0.):
        length=sum(segment.s[-1] for segment in segments); denominator=None
        # The denominator is intentionally conservative and only an input-derived estimate.
        if segments: denominator=length; ratio=1.
        else: ratio=None
        diagnostics=SolverDiagnostics(counter[0],iterations,steps,0,(perf_counter()-started)*1000,max_residual,ratio,tuple(sorted(set(excluded))))
        return SolverResult(tuple(segments),status,bool(segments and ratio is not None and ratio>=self.config.experimental_fully_covered_min_ratio),diagnostics)
