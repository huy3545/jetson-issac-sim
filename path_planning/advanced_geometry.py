"""Verified subsets for Phase 1B; deliberately conservative for closed geometry."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .planar_medial_axis import PlanarMedialAxisSolver, SolverResult, SolverStatus

class GeometryStatus(str, Enum): VERIFIED="VERIFIED"; INVALID="INVALID"; UNCERTAIN="UNCERTAIN"
@dataclass(frozen=True)
class GeometrySegment:
    segment_id: int; input_range: tuple[int,int]; status: GeometryStatus
    solver_result: SolverResult|None; reason_codes: tuple[str,...]=()
@dataclass(frozen=True)
class SegmentedGeometryResult:
    segments: tuple[GeometrySegment,...]; reason_codes: tuple[str,...]=()

def _runs(valid):
    start=None
    for i,ok in enumerate(valid):
        if ok and start is None:start=i
        if start is not None and (not ok or i==len(valid)-1):
            yield start,i if not ok else i+1; start=None

class AdvancedGeometryPlanner:
    def __init__(self, solver=None): self.solver=solver or PlanarMedialAxisSolver()
    def plan_partial(self,left,right,left_valid,right_valid):
        """Solve matching observed runs independently; never bridges invalid indices."""
        left=np.asarray(left,float); right=np.asarray(right,float)
        valid=np.asarray(left_valid,bool)&np.asarray(right_valid,bool)
        segments=[]
        for index,(start,end) in enumerate(_runs(valid)):
            if end-start<2:
                segments.append(GeometrySegment(index,(start,end),GeometryStatus.UNCERTAIN,None,("INSUFFICIENT_OBSERVED_POINTS",))); continue
            result=self.solver.solve(left[start:end],right[start:end])
            status=GeometryStatus.VERIFIED if result.overall_status is SolverStatus.CONVERGED else GeometryStatus.INVALID
            segments.append(GeometrySegment(index,(start,end),status,result,result.diagnostics.excluded))
        if not segments:return SegmentedGeometryResult((),("MISSING_OBSERVATION",))
        return SegmentedGeometryResult(tuple(segments),("PARTIAL_GEOMETRY",) if not valid.all() else ())
    def plan_closed_concentric(self, inner, outer, closure_tol=1e-6):
        """Support only verified concentric circular annuli; no arbitrary seam choice."""
        inner=np.asarray(inner,float); outer=np.asarray(outer,float)
        if len(inner)<4 or len(outer)<4 or np.linalg.norm(inner[0]-inner[-1])>closure_tol or np.linalg.norm(outer[0]-outer[-1])>closure_tol:
            return GeometrySegment(0,(0,0),GeometryStatus.INVALID,None,("INVALID_CLOSED_TOPOLOGY",))
        a,b=inner[:-1],outer[:-1]; ca=a.mean(0); cb=b.mean(0); ra=np.linalg.norm(a-ca,axis=1); rb=np.linalg.norm(b-cb,axis=1)
        if np.linalg.norm(ca-cb)>closure_tol or ra.std()>closure_tol or rb.std()>closure_tol or ra.mean()>=rb.mean():
            return GeometrySegment(0,(0,0),GeometryStatus.UNCERTAIN,None,("CLOSED_CONTOUR_UNSUPPORTED",))
        n=max(len(a),len(b)); angle=np.linspace(0,2*np.pi,n,endpoint=False)
        xy=np.c_[ca[0]+(ra.mean()+rb.mean())/2*np.cos(angle),ca[1]+(ra.mean()+rb.mean())/2*np.sin(angle)]
        xy=np.vstack((xy,xy[0])); s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        from .planar_medial_axis import SolverSegment, SolverDiagnostics
        segment=SolverSegment(xy,s,np.full(len(xy),rb.mean()-ra.mean()),np.ones(len(xy),bool),SolverStatus.CONVERGED,())
        result=SolverResult((segment,),SolverStatus.CONVERGED,True,SolverDiagnostics(coverage_ratio=1.0))
        return GeometrySegment(0,(0,len(xy)),GeometryStatus.VERIFIED,result,())
