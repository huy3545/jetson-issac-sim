"""Pure V1.1 helpers; V1's process() remains the compatibility entry point."""
import numpy as np
from .contracts import OrientationSource

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
