"""Stage 7C-4 analysis metadata for public two-boundary Path results only."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .vision_adapter import VisionPathResult, VisionPathStatus

class CurvatureStatus(str,Enum): AVAILABLE="AVAILABLE"; UNAVAILABLE_AT_CORNER="UNAVAILABLE_AT_CORNER"; UNAVAILABLE="UNAVAILABLE"
class CornerClass(str,Enum): NONE="NONE"; LEFT_TURN="LEFT_TURN"; RIGHT_TURN="RIGHT_TURN"; TURNBACK="TURNBACK"; UNKNOWN="UNKNOWN"

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

@dataclass(frozen=True)
class TwoLineGeometryMetadata:
    arc_length_m:np.ndarray; tangent_xy:np.ndarray; curvature_m_inv:np.ndarray
    curvature_status:CurvatureStatus; left_turn:CornerClass; right_turn:CornerClass
    continuity:bool; coordinate_frame:str; units:str; reason_codes:tuple[str,...]=()
    def __post_init__(self):
        for name in ("arc_length_m","tangent_xy","curvature_m_inv"):
            object.__setattr__(self,name,_ro(getattr(self,name)))

class TwoLineGeometryAnalyzer:
    """Does not alter a centerline or infer a tool/C-axis orientation."""
    def analyze(self,path:VisionPathResult,left,right,*,coordinate_frame,units):
        codes=[]; left_turn=_source_corner(left); right_turn=_source_corner(right)
        if path.status is not VisionPathStatus.VERIFIED or path.solver_result is None or not path.solver_result.segments:
            return TwoLineGeometryMetadata(np.array([]),np.empty((0,2)),np.array([]),CurvatureStatus.UNAVAILABLE,left_turn,right_turn,False,coordinate_frame,units,("PATH_NOT_VERIFIED",)+path.reason_codes)
        xy=path.solver_result.segments[0].xy; ds=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        s=np.r_[0.,np.cumsum(ds)]; continuity=bool(len(ds) and np.all(ds>0) and np.isfinite(xy).all())
        source_corner=left_turn is not CornerClass.NONE or right_turn is not CornerClass.NONE
        if not continuity:
            return TwoLineGeometryMetadata(s,np.full_like(xy,np.nan),np.full(len(xy),np.nan),CurvatureStatus.UNAVAILABLE,left_turn,right_turn,False,coordinate_frame,units,("CENTERLINE_DISCONTINUOUS",))
        tangent=np.gradient(xy,s,axis=0); tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        if source_corner:
            codes.append("CURVATURE_UNAVAILABLE_AT_SOURCE_CORNER")
            curvature=np.full(len(xy),np.nan); status=CurvatureStatus.UNAVAILABLE_AT_CORNER
        else:
            curvature=np.linalg.norm(np.gradient(tangent,s,axis=0),axis=1); status=CurvatureStatus.AVAILABLE
        return TwoLineGeometryMetadata(s,tangent,curvature,status,left_turn,right_turn,True,coordinate_frame,units,tuple(codes))
