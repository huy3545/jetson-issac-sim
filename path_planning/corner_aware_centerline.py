"""Experimental, fail-closed medial-axis solver for one square channel corner.

The supported construction is deliberately narrow: two open, uniformly offset
boundaries with one 90 degree turn and a mitered outer corner.  It is the
geometric medial axis (equal distance to the *entire* boundary polylines), not
the former midpoint/toolpath shortcut.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .planar_medial_axis import (PlanarMedialAxisSolver, SolverDiagnostics,
                                 SolverResult, SolverSegment, SolverStatus)

@dataclass(frozen=True)
class CornerAwareConfig:
    corner_angle_deg:float=30.0
    corner_fraction_tolerance:float=.15
    resample_step_m:float=.005
    experimental_touch_tol_m:float=1e-6
    structure_tolerance_m:float=1e-7
    medial_residual_tolerance_m:float=1e-6
    polyline_verification_subdivisions:int=16
    def __post_init__(self):
        def finite(name,value,*,positive=False,nonnegative=False):
            if isinstance(value,bool) or not isinstance(value,(int,float,np.integer,np.floating)):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value): raise ValueError(f"{name} must be finite")
            if positive and value<=0: raise ValueError(f"{name} must be strictly positive")
            if nonnegative and value<0: raise ValueError(f"{name} must be nonnegative")
        finite("corner_angle_deg",self.corner_angle_deg,positive=True)
        if self.corner_angle_deg>=180: raise ValueError("corner_angle_deg must be < 180")
        finite("corner_fraction_tolerance",self.corner_fraction_tolerance,nonnegative=True)
        if self.corner_fraction_tolerance>1: raise ValueError("corner_fraction_tolerance must be <= 1")
        finite("resample_step_m",self.resample_step_m,positive=True)
        finite("experimental_touch_tol_m",self.experimental_touch_tol_m,nonnegative=True)
        finite("structure_tolerance_m",self.structure_tolerance_m,nonnegative=True)
        finite("medial_residual_tolerance_m",self.medial_residual_tolerance_m,positive=True)
        value=self.polyline_verification_subdivisions
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise TypeError("polyline_verification_subdivisions must be an integer")
        if value<16: raise ValueError("polyline_verification_subdivisions must be at least 16")

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
def _distance(point,line):
    best=float("inf")
    for start,end in zip(line[:-1],line[1:]):
        vector=end-start; denominator=float(vector@vector)
        if denominator<=0: continue
        fraction=float(np.clip(((point-start)@vector)/denominator,0.,1.))
        best=min(best,float(np.linalg.norm(point-(start+fraction*vector))))
    return best

def _unit(vector):
    length=float(np.linalg.norm(vector))
    return None if length<=1e-12 else vector/length

def _left_normal(vector): return np.array([-vector[1],vector[0]])

def _same(a,b,tolerance): return bool(np.linalg.norm(a-b)<=tolerance)

def _append(parts, points):
    points=np.asarray(points,float)
    if len(parts) and len(points) and np.linalg.norm(parts[-1][-1]-points[0])<=1e-12:
        points=points[1:]
    if len(points): parts.append(points)

def _polyline_residual(xy,left,right,subdivisions):
    """Numerically certify every published edge, including its interior."""
    fractions=np.linspace(0.,1.,subdivisions+1)
    points=(xy[:-1,None,:]+fractions[None,:,None]*(xy[1:,None,:]-xy[:-1,None,:])).reshape(-1,2)
    def distances(line):
        start=line[:-1]; vector=line[1:]-start
        denominator=np.einsum("ij,ij->i",vector,vector)
        fraction=np.clip(np.einsum("nmk,mk->nm",points[:,None,:]-start[None,:,:],vector)/denominator[None,:],0.,1.)
        nearest=start[None,:,:]+fraction[:,:,None]*vector[None,:,:]
        return np.linalg.norm(points[:,None,:]-nearest,axis=2).min(axis=1)
    residual=np.abs(distances(left)-distances(right))
    return float(residual.max()),float(np.sqrt(np.mean(residual*residual)))

class CornerAwareTwoLineSolver:
    """Solve a proven square-corner medial axis; reject all other sharp corners."""
    def __init__(self,config=CornerAwareConfig(),base=None): self.config=config;self.base=base or PlanarMedialAxisSolver()
    def solve(self,left,right):
        # Run the existing validator before touching shape-dependent endpoints
        # or attempting traversal normalization.
        try:
            left,right=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
        except (TypeError,ValueError):
            return self.base.solve(left,right)
        baseline=self.base.solve(left,right)
        fatal={"INVALID_SHAPE","NON_FINITE_INPUT","TOO_FEW_VERTICES","TOO_MANY_VERTICES",
               "ZERO_LENGTH_SEGMENT","SELF_INTERSECTION","BOUNDARY_INTERSECTION","BOUNDARY_TOUCH"}
        if baseline.overall_status is SolverStatus.INVALID_GEOMETRY or fatal.intersection(baseline.diagnostics.excluded):
            return baseline
        # Boundary order is a correspondence convention, not a geometric
        # property.  Normalize only the traversal direction before inspecting
        # the turn; boundary identity itself remains left/right.
        if len(left)>=2 and len(right)>=2 and np.linalg.norm(left[0]-right[-1])+np.linalg.norm(left[-1]-right[0]) < np.linalg.norm(left[0]-right[0])+np.linalg.norm(left[-1]-right[-1]):
            right=right[::-1]
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
        # The Phase 1A result is only a validation signal here.  Its marching
        # implementation can legitimately stop at this feature switch, so a
        # prefix must not be mistaken for the complete L-channel axis.
        solved=self._square_corner_axis(left,right,li,ri,ls)
        if solved is None:return self._reject("CORNER_MEDIAL_AXIS_UNVERIFIED")
        return solved

    def _square_corner_axis(self,left,right,li,ri,turn_sign):
        """Return an analytic medial axis for a verified mitered square turn.

        In a local frame at the inner corner, ``a`` is the incoming direction
        and ``n`` points toward the inner boundary's side.  The transition is
        formed by the equal-distance loci of boundary segments/vertices; it
        never uses an array-index midpoint as geometry evidence.
        """
        cfg=self.config
        # This implementation is intentionally restricted to the independently
        # qualified three-vertex synthetic contract.
        if len(left)!=3 or len(right)!=3:return None
        inner,outer=(left,right) if turn_sign>0 else (right,left)
        incoming=_unit(inner[1]-inner[0]); outgoing=_unit(inner[2]-inner[1])
        outer_in=_unit(outer[1]-outer[0]); outer_out=_unit(outer[2]-outer[1])
        if any(item is None for item in (incoming,outgoing,outer_in,outer_out)):return None
        if abs(float(incoming@outgoing))>1e-8:return None
        cross=float(incoming[0]*outgoing[1]-incoming[1]*outgoing[0])
        if abs(abs(cross)-1.)>1e-8:return None
        if float(incoming@outer_in)<1-1e-8 or float(outgoing@outer_out)<1-1e-8:return None
        normal=_left_normal(incoming) if turn_sign>0 else -_left_normal(incoming)
        width=float(np.linalg.norm(outer[0]-inner[0]))
        if width<=cfg.experimental_touch_tol_m:return None
        tolerance=max(cfg.structure_tolerance_m,width*1e-6)
        # A square channel has a common width before/after the corner and its
        # miter vertex is fixed by the two offset line directions.
        if not _same(outer[0],inner[0]-width*normal,tolerance):return None
        post_normal=_left_normal(outgoing) if turn_sign>0 else -_left_normal(outgoing)
        if not _same(outer[2],inner[2]-width*post_normal,tolerance):return None
        expected_outer_corner=inner[1]+width*incoming-width*normal
        if not _same(outer[1],expected_outer_corner,tolerance):return None
        if min(_length(inner[:2]),_length(inner[1:]),_length(outer[:2]),_length(outer[1:]))<=width/2:return None

        # Axis: incoming straight section; segment/vertex parabola; vertex/
        # outer-segment locus; outgoing straight section.  q is distance along
        # incoming a, r along the signed inner normal, both relative to the
        # inner corner.  The join q/W=sqrt(2)-1 follows from equality of the
        # active boundary features.
        q_join=width*(np.sqrt(2.)-1.)
        # Output resolution is an upper bound, not a minimum: a larger caller
        # resample step must not weaken the continuous residual certificate.
        fine=min(cfg.resample_step_m/10.,width/400.)
        def count(length): return max(2,int(np.ceil(length/fine))+1)
        parts=[]
        start=inner[0]-width/2*normal
        corner=inner[1]
        end=inner[2]-width/2*post_normal
        _append(parts,np.linspace(start,corner-width/2*normal,count(float(np.linalg.norm((corner-width/2*normal)-start)))))
        q=np.linspace(0.,q_join,count(q_join))
        r=-width/2+q*q/(2*width)
        _append(parts,corner+q[:,None]*incoming+r[:,None]*normal)
        # Parameterize this branch by r rather than q.  q spacing collapses
        # near its vertical tangent and would make the published polyline a
        # poor representation between samples.
        r_join=-np.sqrt(width*width-2*width*q_join)
        r=np.linspace(r_join,0.,count(abs(r_join)))
        q=width/2-r*r/(2*width)
        _append(parts,corner+q[:,None]*incoming+r[:,None]*normal)
        transition_end=corner+width/2*incoming
        _append(parts,np.linspace(transition_end,end,count(float(np.linalg.norm(end-transition_end)))))
        xy=np.vstack(parts)
        ds=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        if len(xy)<2 or not np.all(ds>0) or not np.isfinite(xy).all():return None
        left_distance=np.array([_distance(point,left) for point in xy])
        right_distance=np.array([_distance(point,right) for point in xy])
        residual=np.abs(left_distance-right_distance)
        continuous_max,continuous_rms=_polyline_residual(xy,left,right,cfg.polyline_verification_subdivisions)
        if max(float(residual.max()),continuous_max)>cfg.medial_residual_tolerance_m:return None
        width_values=left_distance+right_distance
        s=np.r_[0.,np.cumsum(ds)]
        # Each construction feature is a separate smooth branch.  Analyzer
        # derivatives must not cross a feature switch, even when the tangent
        # happens to be continuous there.
        q0_point=corner+q_join*incoming+(-width/2+q_join*q_join/(2*width))*normal
        q1_point=corner+width/2*incoming
        q_start=corner-width/2*normal
        transitions=tuple(sorted({int(np.argmin(np.linalg.norm(xy-point,axis=1))) for point in (q_start,q0_point,q1_point)}))
        cuts=(0,)+transitions+(len(xy)-1,)
        ranges=tuple((cuts[index],cuts[index+1]) for index in range(len(cuts)-1) if cuts[index+1]-cuts[index]>=2)
        segment=SolverSegment(xy,s,width_values,np.ones(len(xy),bool),SolverStatus.CONVERGED,("ANALYTIC_SQUARE_CORNER_MEDIAL_AXIS",),ranges,transitions)
        diagnostics=SolverDiagnostics(max_residual_m=continuous_max,coverage_ratio=1.,excluded=("ANALYTIC_SQUARE_CORNER_MEDIAL_AXIS",),coverage_estimation_method=f"analytic square-corner medial axis; endpoint-to-endpoint coverage; {cfg.polyline_verification_subdivisions} interior subdivisions per output edge; residual_rms_m={continuous_rms:.17g}")
        return SolverResult((segment,),SolverStatus.CONVERGED,True,diagnostics)
    def _reject(self,code):
        return SolverResult((),SolverStatus.AMBIGUOUS,False,SolverDiagnostics(excluded=(code,)))
