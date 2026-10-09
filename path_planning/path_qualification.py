"""Phase 3 immutable 3D path handoff; contains no motion commands."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .surface_mapping import SurfaceStatus

class PathQualificationStatus(str,Enum): VERIFIED='VERIFIED'; INVALID='INVALID'; UNCERTAIN='UNCERTAIN'
class UncertaintyStatus(str,Enum): KNOWN='KNOWN'; UNKNOWN='UNKNOWN'; EXCEEDED='EXCEEDED'
@dataclass(frozen=True)
class PathQualificationConfig:
 policy:str='strict'; experimental_max_3d_step_m:float|None=None; experimental_max_z_jump_m:float|None=None; experimental_max_surface_uncertainty_m:float|None=None
@dataclass(frozen=True)
class QualifiedPathSegment:
 segment_id:str; source_id:int; child_id:int; source_indices:tuple[int,int]; xyz:np.ndarray; s_m:np.ndarray; tangent_xyz:np.ndarray; curvature_m_inv:np.ndarray; uncertainty_status:UncertaintyStatus; status:PathQualificationStatus; coordinate_frame:str; units:str; timestamp:float; clock_domain:str; reason_codes:tuple[str,...]=()
@dataclass(frozen=True)
class QualifiedPathResult: schema_version:str; segments:tuple[QualifiedPathSegment,...]; reason_codes:tuple[str,...]=()
def _ro(a):
 b=np.array(a,dtype=float,copy=True);b.setflags(write=False);return b
def qualify_surface_segments(surface_result,*,coordinate_frame,units,timestamp,clock_domain,config=PathQualificationConfig(),uncertainty=None):
 if config.policy not in ('strict','geometry_only'): raise ValueError('unknown qualification policy')
 out=[]; reasons=[]
 for child,seg in enumerate(surface_result.segments):
  xyz=np.asarray(seg.xyz,float); code=[]; status=PathQualificationStatus.VERIFIED
  if seg.status is not SurfaceStatus.VERIFIED: code.append('UNVERIFIED_GEOMETRY')
  if units!='m' or not coordinate_frame or not clock_domain or not np.isfinite(timestamp): code.append('INVALID_PROVENANCE')
  if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all(): code.append('INVALID_XYZ')
  ds=np.linalg.norm(np.diff(xyz,axis=0),axis=1) if xyz.ndim==2 and len(xyz)>1 else np.array([])
  if len(ds) and (ds<=0).any(): code.append('CONSECUTIVE_DUPLICATE')
  if config.experimental_max_3d_step_m is not None and len(ds) and (ds>config.experimental_max_3d_step_m).any():code.append('STEP_EXCEEDED')
  if config.experimental_max_z_jump_m is not None and len(xyz)>1 and (np.abs(np.diff(xyz[:,2]))>config.experimental_max_z_jump_m).any():code.append('Z_JUMP_EXCEEDED')
  if uncertainty is not None and (not np.isfinite(uncertainty).all()): code.append('INVALID_UNCERTAINTY')
  us=UncertaintyStatus.UNKNOWN if uncertainty is None else UncertaintyStatus.KNOWN
  if config.policy=='strict' and (uncertainty is None or config.experimental_max_surface_uncertainty_m is None):code.append('UNCERTAINTY_UNCONFIGURED')
  if uncertainty is not None and config.experimental_max_surface_uncertainty_m is not None and np.nanmax(uncertainty)>config.experimental_max_surface_uncertainty_m:code.append('SURFACE_UNCERTAINTY_EXCEEDED');us=UncertaintyStatus.EXCEEDED
  if code:status=PathQualificationStatus.UNCERTAIN if any('UNCERTAINTY' in c for c in code) else PathQualificationStatus.INVALID
  s=np.r_[0,np.cumsum(ds)] if len(ds) else np.zeros(len(xyz)); tangent=np.gradient(xyz,s,axis=0) if len(ds) and np.all(np.diff(s)>0) else np.full_like(xyz,np.nan); curvature=np.linalg.norm(np.gradient(tangent,s,axis=0),axis=1) if np.isfinite(tangent).all() else np.full(len(xyz),np.nan)
  out.append(QualifiedPathSegment(f'{seg.source_id}:{child}',seg.source_id,child,seg.input_range,_ro(xyz),_ro(s),_ro(tangent),_ro(curvature),us,status,coordinate_frame,units,timestamp,clock_domain,tuple(code)))
 return QualifiedPathResult('1.0',tuple(out),tuple(reasons))
