"""Pure V1.1 helpers; V1's process() remains the compatibility entry point."""
from dataclasses import dataclass
import numpy as np
from .contracts import OrientationSource

@dataclass(frozen=True)
class ResampleSegmentReference:
    start_index:int|None; end_index:int|None; fraction:float|None; available:bool

@dataclass(frozen=True)
class ResampledPixelGeometry:
    points:np.ndarray
    lineage:tuple[ResampleSegmentReference,...]
    omitted_corner_indices:tuple[int,...]=()
    corner_status:str="PRESERVED"
    def __post_init__(self):
        array=np.ascontiguousarray(np.array(self.points,dtype=float,copy=True))
        object.__setattr__(self,"points",np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape))

def _interior_corners(points, closed):
    stop=len(points) if closed else len(points)-1
    corners=[]
    for index in range(1,stop):
        before=points[index]-points[index-1]; after=points[(index+1)%len(points)]-points[index]
        if np.linalg.norm(before)>1e-12 and np.linalg.norm(after)>1e-12:
            cosine=np.clip(np.dot(before,after)/(np.linalg.norm(before)*np.linalg.norm(after)),-1.,1.)
            if not np.isclose(np.arccos(cosine),0.,atol=1e-10): corners.append(index)
    return corners

def resample_arc_length_with_lineage(points:np.ndarray,count:int,*,closed:bool=False)->ResampledPixelGeometry:
    """Return legacy-equivalent samples plus exact source segment/fraction evidence.

    Zero-length-only input has no deterministic supporting segment and is returned
    with unavailable lineage rather than manufactured provenance.
    """
    source=np.asarray(points,dtype=float)
    if source.ndim!=2 or source.shape[1:]!=(2,) or len(source)==0 or not np.isfinite(source).all() or count<1:
        raise ValueError("points must be finite Nx2 and count must be positive")
    work=source if not closed or np.array_equal(source[0],source[-1]) else np.vstack((source,source[0]))
    segment_lengths=np.linalg.norm(np.diff(work,axis=0),axis=1)
    distances=np.r_[0.,np.cumsum(segment_lengths)]
    samples=np.linspace(0.,distances[-1],count)
    output=np.column_stack([np.interp(samples,distances,work[:,axis]) for axis in range(2)])
    if distances[-1]<=1e-12:
        lineage=tuple(ResampleSegmentReference(None,None,None,False) for _ in output)
        return ResampledPixelGeometry(output,lineage,(),"UNAVAILABLE")
    lineage=[]
    for distance in samples:
        index=int(np.searchsorted(distances,distance,side="right")-1)
        index=min(max(index,0),len(segment_lengths)-1)
        while index<len(segment_lengths)-1 and segment_lengths[index]<=1e-12: index+=1
        length=segment_lengths[index]
        if length<=1e-12:
            lineage.append(ResampleSegmentReference(None,None,None,False)); continue
        fraction=float(np.clip((distance-distances[index])/length,0.,1.))
        lineage.append(ResampleSegmentReference(index,(index+1)%len(source),fraction,True))
    # A non-collinear original vertex must occur verbatim in samples to claim it
    # was preserved.  The geometry is still returned for compatibility.
    corners=_interior_corners(source,closed)
    omitted=tuple(index for index in corners if not np.any(np.all(np.isclose(output,source[index],atol=1e-10),axis=1)))
    return ResampledPixelGeometry(output,tuple(lineage),omitted,"LIMITED" if omitted else "PRESERVED")

def resample_arc_length(points: np.ndarray, count: int) -> np.ndarray:
    points=np.asarray(points,dtype=float); dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    sample=np.linspace(0,dist[-1],count)
    return np.column_stack([np.interp(sample,dist,points[:,i]) for i in range(2)])

def sample_depth_valid(depth: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float, tuple[tuple[int,int],...]]:
    u=np.rint(points[:,0]).astype(int); v=np.rint(points[:,1]).astype(int)
    valid=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
    values=np.zeros(len(points)); values[valid]=depth[v[valid],u[valid]]
    valid &= np.isfinite(values)&(values>0)
    runs=[]; start=None
    for i, ok in enumerate(valid):
        if ok and start is None: start=i
        if start is not None and (not ok or i==len(valid)-1):
            runs.append((start,i if not ok else i+1)); start=None
    return valid, float(valid.mean()) if len(valid) else 0.0, tuple(runs)

def semantic_order(a: np.ndarray, b: np.ndarray, c_axis: float | None=None, reference_left: np.ndarray | None=None):
    """Order by path normal; only fall back to C axis then reference relation."""
    tangent=np.mean(np.gradient((a+b)/2,axis=0),axis=0)
    if np.linalg.norm(tangent)>1e-8:
        normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
        if abs(np.dot(np.mean(b,axis=0)-np.mean(a,axis=0),normal))>1e-6:
            return (a,b,OrientationSource.GEOMETRIC) if np.dot(np.mean(b,axis=0)-np.mean(a,axis=0),normal)>0 else (b,a,OrientationSource.GEOMETRIC)
    if c_axis is not None:
        normal=np.array([-np.sin(c_axis),np.cos(c_axis)])
        return (a,b,OrientationSource.C_AXIS) if np.dot(np.mean(b,axis=0)-np.mean(a,axis=0),normal)>0 else (b,a,OrientationSource.C_AXIS)
    if reference_left is not None:
        return (a,b,OrientationSource.REFERENCE) if np.linalg.norm(a.mean(0)-reference_left.mean(0)) <= np.linalg.norm(b.mean(0)-reference_left.mean(0)) else (b,a,OrientationSource.REFERENCE)
    return None,None,OrientationSource.AMBIGUOUS
