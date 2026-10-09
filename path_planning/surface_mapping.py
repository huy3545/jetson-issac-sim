"""Phase 2 metric heightmap mapping. No cutter offset or implicit transform."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .advanced_geometry import GeometrySegment, GeometryStatus

class SurfaceStatus(str, Enum): VERIFIED="VERIFIED"; INVALID="INVALID"; UNCERTAIN="UNCERTAIN"
@dataclass(frozen=True)
class MetricHeightmap:
    z: np.ndarray; origin_xy: tuple[float,float]; spacing_xy: tuple[float,float]
    coordinate_frame: str; units: str; timestamp: float; clock_domain: str="simulation"
    valid_mask: np.ndarray|None=None; uncertainty: np.ndarray|float|None=None
@dataclass(frozen=True)
class SurfaceSegment:
    source_id:int; xyz:np.ndarray; status:SurfaceStatus; input_range:tuple[int,int]; reason_codes:tuple[str,...]=()
@dataclass(frozen=True)
class SurfaceMappingResult:
    segments:tuple[SurfaceSegment,...]; reason_codes:tuple[str,...]=()

class SurfaceMapper:
 def __init__(self, experimental_max_uncertainty=None, experimental_stale_ms=250.): self.max_uncertainty=experimental_max_uncertainty;self.stale=experimental_stale_ms/1000
 def _sample(self,h,p):
  if h.units!='m' or not h.coordinate_frame or not np.isfinite(h.timestamp):return None,'INVALID_HEIGHTMAP'
  z=np.asarray(h.z,float); mask=np.ones(z.shape,bool) if h.valid_mask is None else np.asarray(h.valid_mask,bool)
  if z.ndim!=2 or mask.shape!=z.shape or min(h.spacing_xy)<=0:return None,'INVALID_HEIGHTMAP'
  u=(p[0]-h.origin_xy[0])/h.spacing_xy[0];v=(p[1]-h.origin_xy[1])/h.spacing_xy[1];i,j=int(np.floor(u)),int(np.floor(v))
  if i<0 or j<0 or i+1>=z.shape[1] or j+1>=z.shape[0]:return None,'SURFACE_OUT_OF_BOUNDS'
  vals=np.array([z[j,i],z[j,i+1],z[j+1,i],z[j+1,i+1]]); valid=np.array([mask[j,i],mask[j,i+1],mask[j+1,i],mask[j+1,i+1]])
  if not valid.all() or not np.isfinite(vals).all():return None,'MISSING_SURFACE_DATA'
  a,b=u-i,v-j; value=vals[0]*(1-a)*(1-b)+vals[1]*a*(1-b)+vals[2]*(1-a)*b+vals[3]*a*b
  if h.uncertainty is None and self.max_uncertainty is not None:return None,'SURFACE_UNCERTAINTY_EXCEEDED'
  if h.uncertainty is not None:
   unc=float(h.uncertainty) if np.ndim(h.uncertainty)==0 else float(np.max(np.asarray(h.uncertainty)[[j,j,j+1,j+1],[i,i+1,i,i+1]]))
   if self.max_uncertainty is not None and unc>self.max_uncertainty:return None,'SURFACE_UNCERTAINTY_EXCEEDED'
  return value,None
 def map_segments(self,segments,heightmap,frame,timestamp,clock_domain='simulation'):
  if frame!=heightmap.coordinate_frame:return SurfaceMappingResult((),('SURFACE_FRAME_MISMATCH',))
  if clock_domain!=heightmap.clock_domain or abs(timestamp-heightmap.timestamp)>self.stale:return SurfaceMappingResult((),('SURFACE_TIMESTAMP_MISMATCH',))
  out=[]
  for segment in segments:
   if segment.status is not GeometryStatus.VERIFIED or segment.solver_result is None:
    out.append(SurfaceSegment(segment.segment_id,np.empty((0,3)),SurfaceStatus.INVALID,segment.input_range,('UNVERIFIED_GEOMETRY',)));continue
   xy=segment.solver_result.segments[0].xy; runs=[]; current=[]; reasons=[]
   for p in xy:
    value,reason=self._sample(heightmap,p)
    if reason is None: current.append([p[0],p[1],value])
    else:
     reasons.append(reason)
     if len(current)>=2:runs.append(np.array(current));current=[]
   if len(current)>=2:runs.append(np.array(current))
   if not runs:out.append(SurfaceSegment(segment.segment_id,np.empty((0,3)),SurfaceStatus.INVALID,segment.input_range,tuple(sorted(set(reasons)))));continue
   for k,xyz in enumerate(runs):out.append(SurfaceSegment(segment.segment_id,xyz,SurfaceStatus.VERIFIED,segment.input_range,('PARTIAL_SURFACE_MAPPING',) if reasons else ()))
  return SurfaceMappingResult(tuple(out))
