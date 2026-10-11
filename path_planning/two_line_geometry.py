"""Stage 7C-4 analysis metadata for public two-boundary Path results only."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .vision_adapter import VisionPathResult, VisionPathStatus

class CurvatureStatus(str,Enum): AVAILABLE="AVAILABLE"; UNAVAILABLE_AT_CORNER="UNAVAILABLE_AT_CORNER"; UNAVAILABLE="UNAVAILABLE"
class CornerClass(str,Enum): NONE="NONE"; LEFT_TURN="LEFT_TURN"; RIGHT_TURN="RIGHT_TURN"; TURNBACK="TURNBACK"; UNKNOWN="UNKNOWN"
@dataclass(frozen=True)
class BoundarySourceReference:
    boundary_id:str; start_index:int|None; end_index:int|None; fraction:float|None; available:bool
@dataclass(frozen=True)
class PointSourceCorrespondence:
    left:BoundarySourceReference; right:BoundarySourceReference; derivation_method:str="GEOMETRIC_SEGMENT_PROJECTION"

def _ro(value):
    array=np.ascontiguousarray(np.array(value,dtype=float,copy=True)); return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)

def _source_corner(points):
    points=np.asarray(points,float); turns=[]
    for index in range(1,len(points)-1):
        before=points[index]-points[index-1]; after=points[index+1]-points[index]
        if np.linalg.norm(before)<=1e-12 or np.linalg.norm(after)<=1e-12: return CornerClass.UNKNOWN
        cross=before[0]*after[1]-before[1]*after[0]; dot=float(before@after)
        angle=np.degrees(np.arccos(np.clip(dot/(np.linalg.norm(before)*np.linalg.norm(after)),-1.,1.)))
        # Sampling a smooth curve has small direction increments; it is not a
        # discontinuous corner merely because the tangent changes.
        if angle<15.: continue
        if np.isclose(cross,0.) and dot<0: turns.append(CornerClass.TURNBACK)
        elif cross>0: turns.append(CornerClass.LEFT_TURN)
        elif cross<0: turns.append(CornerClass.RIGHT_TURN)
    return turns[0] if turns else CornerClass.NONE
def _closest_reference(point,line,boundary_id):
    line=np.asarray(line,float); choice=None
    for index,(start,end) in enumerate(zip(line[:-1],line[1:])):
        vector=end-start; denominator=float(vector@vector)
        if denominator<=0: continue
        fraction=float(np.clip(((point-start)@vector)/denominator,0.,1.)); distance=float(np.linalg.norm(point-(start+fraction*vector)))
        if choice is None or distance<choice[0]: choice=(distance,index,fraction)
    if choice is None:return BoundarySourceReference(boundary_id,None,None,None,False)
    _,index,fraction=choice;return BoundarySourceReference(boundary_id,index,index+1,fraction,True)
def _corner_index(xy):
    if len(xy)<3:return None
    angles=[]
    for index in range(1,len(xy)-1):
        a=xy[index]-xy[index-1];b=xy[index+1]-xy[index];na=np.linalg.norm(a);nb=np.linalg.norm(b)
        angles.append(0. if na<=1e-12 or nb<=1e-12 else float(np.degrees(np.arccos(np.clip((a@b)/(na*nb),-1.,1.)))))
    index=int(np.argmax(angles));return index+1 if angles[index]>=30. else None

def _branch_derivatives(xy,s,ranges):
    """Central differences confined to declared smooth feature branches."""
    tangent=np.full_like(xy,np.nan); curvature=np.full(len(xy),np.nan)
    tangent_valid=np.zeros(len(xy),bool); curvature_valid=np.zeros(len(xy),bool)
    for start,end in ranges:
        if start<0 or end>=len(xy) or end-start<4: continue
        for index in range(start+1,end):
            delta_s=s[index+1]-s[index-1]
            vector=(xy[index+1]-xy[index-1])/delta_s
            norm=np.linalg.norm(vector)
            if norm>1e-12:
                tangent[index]=vector/norm; tangent_valid[index]=True
        for index in range(start+2,end-1):
            if not tangent_valid[index-1] or not tangent_valid[index+1]: continue
            delta_s=s[index+1]-s[index-1]
            curvature[index]=np.linalg.norm((tangent[index+1]-tangent[index-1])/delta_s)
            curvature_valid[index]=np.isfinite(curvature[index])
    # The public metadata has one validity contract for differential geometry:
    # a tangent is not published where its curvature stencil is unavailable.
    tangent[~curvature_valid]=np.nan
    tangent_valid=curvature_valid.copy()
    return tangent,curvature,tangent_valid,curvature_valid

@dataclass(frozen=True)
class TwoLineGeometryMetadata:
    arc_length_m:np.ndarray; tangent_xy:np.ndarray; curvature_m_inv:np.ndarray
    curvature_status:CurvatureStatus; left_turn:CornerClass; right_turn:CornerClass
    continuity:bool; coordinate_frame:str; units:str; reason_codes:tuple[str,...]=()
    source_indices:np.ndarray|None=None; curvature_valid:np.ndarray|None=None
    source_correspondence:tuple[PointSourceCorrespondence,...]=()
    tangent_valid:np.ndarray|None=None
    def __post_init__(self):
        for name in ("arc_length_m","tangent_xy","curvature_m_inv"):
            object.__setattr__(self,name,_ro(getattr(self,name)))
        count=len(self.arc_length_m)
        # Output order is not source provenance.  -1 explicitly means unavailable
        # for the legacy field; exact boundary references live below.
        indices=np.full(count,-1,dtype=int) if self.source_indices is None else np.asarray(self.source_indices,dtype=int)
        valid=np.zeros(count,dtype=bool) if self.curvature_valid is None else np.asarray(self.curvature_valid,dtype=bool)
        tangent_valid=np.isfinite(self.tangent_xy).all(axis=1) if self.tangent_valid is None else np.asarray(self.tangent_valid,dtype=bool)
        indices=np.frombuffer(np.ascontiguousarray(indices).tobytes(),dtype=indices.dtype).reshape(indices.shape)
        valid=np.frombuffer(np.ascontiguousarray(valid).tobytes(),dtype=valid.dtype).reshape(valid.shape)
        tangent_valid=np.frombuffer(np.ascontiguousarray(tangent_valid).tobytes(),dtype=tangent_valid.dtype).reshape(tangent_valid.shape)
        object.__setattr__(self,"source_indices",indices);object.__setattr__(self,"curvature_valid",valid);object.__setattr__(self,"tangent_valid",tangent_valid)
        if self.source_correspondence and len(self.source_correspondence)!=count: raise ValueError("source_correspondence must align with output")

class TwoLineGeometryAnalyzer:
    """Does not alter a centerline or infer a tool/C-axis orientation."""
    def analyze(self,path:VisionPathResult,left,right,*,coordinate_frame,units):
        codes=[]; left_turn=_source_corner(left); right_turn=_source_corner(right)
        if path.status is not VisionPathStatus.VERIFIED or path.solver_result is None or not path.solver_result.segments:
            return TwoLineGeometryMetadata(np.array([]),np.empty((0,2)),np.array([]),CurvatureStatus.UNAVAILABLE,left_turn,right_turn,False,coordinate_frame,units,("PATH_NOT_VERIFIED",)+path.reason_codes)
        xy=path.solver_result.segments[0].xy; ds=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        s=np.r_[0.,np.cumsum(ds)]; continuity=bool(len(ds) and np.all(ds>0) and np.isfinite(xy).all())
        source_corner=left_turn is not CornerClass.NONE or right_turn is not CornerClass.NONE
        correspondence=tuple(PointSourceCorrespondence(_closest_reference(point,left,"left"),_closest_reference(point,right,"right")) for point in xy)
        if not continuity:
            return TwoLineGeometryMetadata(s,np.full_like(xy,np.nan),np.full(len(xy),np.nan),CurvatureStatus.UNAVAILABLE,left_turn,right_turn,False,coordinate_frame,units,("CENTERLINE_DISCONTINUOUS",),None,np.zeros(len(xy),bool),correspondence)
        segment=path.solver_result.segments[0]
        branches=getattr(segment,"smooth_branch_ranges",())
        if branches:
            tangent,curvature,tangent_valid,valid=_branch_derivatives(xy,s,branches)
            # Transition and adjacent stencil points are intentionally NaN.
            status=CurvatureStatus.UNAVAILABLE_AT_CORNER
            codes.append("CURVATURE_LOCALIZED_BY_SMOOTH_BRANCH")
            return TwoLineGeometryMetadata(s,tangent,curvature,status,left_turn,right_turn,True,coordinate_frame,units,tuple(codes),None,valid,correspondence,tangent_valid)
        tangent=np.gradient(xy,s,axis=0); tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        if source_corner:
            codes.append("CURVATURE_UNAVAILABLE_AT_SOURCE_CORNER")
            curvature=np.linalg.norm(np.gradient(tangent,s,axis=0),axis=1); valid=np.isfinite(curvature)
            index=_corner_index(xy)
            if index is None:
                return TwoLineGeometryMetadata(s,tangent,np.full(len(xy),np.nan),CurvatureStatus.UNAVAILABLE,left_turn,right_turn,True,coordinate_frame,units,("CORNER_POSITION_UNRESOLVED",),None,np.zeros(len(xy),bool),correspondence)
            curvature[index]=np.nan;tangent[index]=np.nan;valid[index]=False;status=CurvatureStatus.UNAVAILABLE_AT_CORNER
        else:
            curvature=np.linalg.norm(np.gradient(tangent,s,axis=0),axis=1); valid=np.isfinite(curvature); status=CurvatureStatus.AVAILABLE
        return TwoLineGeometryMetadata(s,tangent,curvature,status,left_turn,right_turn,True,coordinate_frame,units,tuple(codes),None,valid,correspondence)
