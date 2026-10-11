"""Experimental dense covariance for small, explicitly ordered point sets."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from .full_covariance import compose_full_jacobian
from .uncertainty_artifact import ArtifactTrust
from .experimental_metric_projection import MetricGeometryStatus,MetricUncertaintyStatus

def _ro(v):a=np.array(v,float,copy=True);a.setflags(write=False);return a
def canonical_ordering(ids):
    return tuple(f"{pid}.{name}" for pid in ids for name in ("u","v","z00","z10","z01","z11"))+("shared.fx","shared.fy","shared.cx","shared.cy","shared.tx","shared.ty","shared.tz","shared.rx","shared.ry","shared.rz")

@dataclass(frozen=True)
class CrossPointEvidence:
    artifact_id:str;artifact_version:str;point_ids:tuple[str,...];point_metadata:tuple[tuple[tuple[str,object],...],...]
    variable_ordering:tuple[str,...];global_input_covariance:np.ndarray;camera_frame:str;reference_frame:str;units:str
    timestamp:float;clock_domain:str;calibration_id:str;calibration_version:str;transform_id:str;transform_version:str
    shared_uncertainty_source_id:str;correlation_source:str;correlation_method:str
    def __post_init__(self):object.__setattr__(self,"global_input_covariance",_ro(self.global_input_covariance))
@dataclass(frozen=True)
class CrossPointArtifactResult:
    trust:ArtifactTrust;diagnostics:tuple[str,...];evidence:CrossPointEvidence|None;experimental:bool=True;execution_authorized:bool=False
@dataclass(frozen=True)
class CrossPointNominal:
    point_id:str;pixel:np.ndarray;depth_samples:np.ndarray;reference_xyz:np.ndarray
    def __post_init__(self):
        object.__setattr__(self,"pixel",_ro(self.pixel));object.__setattr__(self,"depth_samples",_ro(self.depth_samples));object.__setattr__(self,"reference_xyz",_ro(self.reference_xyz))
@dataclass(frozen=True)
class CrossPointBlock:
    first_point_id:str;second_point_id:str;covariance:np.ndarray
    def __post_init__(self):object.__setattr__(self,"covariance",_ro(self.covariance))
@dataclass(frozen=True)
class CrossPointCovarianceResult:
    point_ids:tuple[str,...];global_xyz_covariance:np.ndarray|None;diagonal_blocks:tuple[CrossPointBlock,...];cross_blocks:tuple[CrossPointBlock,...]
    geometry_status:MetricGeometryStatus;uncertainty_status:MetricUncertaintyStatus;diagnostics:tuple[str,...]
    experimental:bool=True;execution_authorized:bool=False
    def __post_init__(self):
        if self.global_xyz_covariance is not None:object.__setattr__(self,"global_xyz_covariance",_ro(self.global_xyz_covariance))

class CrossPointArtifactIngestor:
    SCHEMA="1.2-cross-point";MAX_POINTS=32;MAX_BYTES=2_000_000
    def ingest(self,raw,expected_digest=None):
        if not isinstance(raw,(bytes,bytearray)):return self._bad("ARTIFACT_BYTES_REQUIRED")
        if len(raw)>self.MAX_BYTES:return self._bad("ARTIFACT_TOO_LARGE")
        if expected_digest is not None and hashlib.sha256(raw).hexdigest()!=expected_digest:return self._bad("ARTIFACT_DIGEST_MISMATCH")
        try:d=json.loads(bytes(raw).decode())
        except (UnicodeDecodeError,json.JSONDecodeError):return self._bad("MALFORMED_JSON")
        reasons=[];ids=d.get("point_ordering")
        if d.get("schema_version")!=self.SCHEMA:reasons.append("UNSUPPORTED_SCHEMA_VERSION")
        if not isinstance(ids,list) or len(ids)<2:reasons.append("MINIMUM_TWO_POINTS_REQUIRED");ids=[]
        if len(ids)>self.MAX_POINTS:return self._bad("POINT_LIMIT_EXCEEDED")
        if len(ids)!=len(set(ids)):reasons.append("DUPLICATE_POINT_IDS")
        points=d.get("points")
        if not isinstance(points,list) or [p.get("point_id") for p in points if isinstance(p,dict)]!=ids:reasons.append("POINT_ORDERING_MISMATCH");points=[]
        if d.get("global_variable_ordering")!=list(canonical_ordering(ids)):reasons.append("VARIABLE_ORDERING_MISMATCH")
        shared=d.get("shared_provenance") if isinstance(d.get("shared_provenance"),dict) else {}
        for key in ("calibration_id","calibration_version","transform_id","transform_version","shared_uncertainty_source_id"):
            if not shared.get(key):reasons.append("MISSING_SHARED_"+key.upper())
        for key in ("camera_frame","reference_frame","units","timestamp","clock_domain"):
            if d.get(key) in (None,""):reasons.append("MISSING_"+key.upper())
        if d.get("units")!="m":reasons.append("UNITS_MISMATCH")
        corr=d.get("correlation_provenance") if isinstance(d.get("correlation_provenance"),dict) else {}
        if not corr.get("source") or not corr.get("method") or corr.get("status")!="PROVIDED":reasons.append("UNKNOWN_CROSS_POINT_CORRELATION")
        for p in points:
            for key in ("point_id","segment_id","source_id","camera_frame","reference_frame","timestamp","clock_domain","calibration_id","calibration_version","transform_id","transform_version","shared_uncertainty_source_id"):
                if p.get(key) in (None,""):reasons.append("POINT_METADATA_INCOMPLETE")
            matches=(("camera_frame","camera_frame"),("reference_frame","reference_frame"),("timestamp","timestamp"),("clock_domain","clock_domain"))
            if any(p.get(a)!=d.get(b) for a,b in matches):reasons.append("POINT_FRAME_TIME_MISMATCH")
            if any(p.get(k)!=shared.get(k) for k in ("calibration_id","calibration_version","transform_id","transform_version","shared_uncertainty_source_id")):reasons.append("POINT_SHARED_PROVENANCE_MISMATCH")
        dim=6*len(ids)+10
        try:C=np.asarray(d.get("global_input_covariance"),float)
        except (TypeError,ValueError):C=np.empty((0,));reasons.append("GLOBAL_COVARIANCE_INVALID")
        if C.shape!=(dim,dim):reasons.append("GLOBAL_COVARIANCE_SHAPE")
        elif not np.isfinite(C).all():reasons.append("GLOBAL_COVARIANCE_NONFINITE")
        elif not np.allclose(C,C.T,atol=1e-12,rtol=0.):reasons.append("GLOBAL_COVARIANCE_NONSYMMETRIC")
        elif np.any(np.diag(C)<0) or np.linalg.eigvalsh(C).min() < -1e-10:reasons.append("GLOBAL_COVARIANCE_NOT_PSD")
        if reasons:return CrossPointArtifactResult(ArtifactTrust.INVALID,tuple(sorted(set(reasons))),None)
        meta=tuple(tuple(sorted(p.items())) for p in points)
        ev=CrossPointEvidence(d["artifact_id"],d["artifact_version"],tuple(ids),meta,canonical_ordering(ids),C,d["camera_frame"],d["reference_frame"],d["units"],d["timestamp"],d["clock_domain"],shared["calibration_id"],shared["calibration_version"],shared["transform_id"],shared["transform_version"],shared["shared_uncertainty_source_id"],corr["source"],corr["method"])
        return CrossPointArtifactResult(ArtifactTrust.EVIDENCE_VALIDATED if expected_digest else ArtifactTrust.DECLARED,(),ev)
    @staticmethod
    def _bad(code):return CrossPointArtifactResult(ArtifactTrust.INVALID,(code,),None)

class ExperimentalCrossPointPropagator:
    def propagate(self,evidence,nominals,calibration,rotation):
        if not isinstance(evidence,CrossPointEvidence) or tuple(n.point_id for n in nominals)!=getattr(evidence,"point_ids",()):return self._bad((),"NOMINAL_POINT_ORDERING_MISMATCH")
        n=len(nominals);J=np.zeros((3*n,6*n+10))
        for i,item in enumerate(nominals):
            local=compose_full_jacobian(item.pixel,item.depth_samples,calibration,rotation,item.reference_xyz);rows=slice(3*i,3*i+3);cols=6*i
            J[rows,cols:cols+6]=local[:,:6];J[rows,6*n:6*n+4]=local[:,6:10];J[rows,6*n+4:]=local[:,10:]
        C=J@evidence.global_input_covariance@J.T;C=(C+C.T)/2
        if not np.isfinite(C).all() or np.linalg.eigvalsh(C).min() < -1e-9:return self._bad(evidence.point_ids,"OUTPUT_COVARIANCE_INVALID")
        diagonal=tuple(CrossPointBlock(pid,pid,C[3*i:3*i+3,3*i:3*i+3]) for i,pid in enumerate(evidence.point_ids))
        cross=tuple(CrossPointBlock(evidence.point_ids[i],evidence.point_ids[j],C[3*i:3*i+3,3*j:3*j+3]) for i in range(n) for j in range(i+1,n))
        return CrossPointCovarianceResult(evidence.point_ids,C,diagonal,cross,MetricGeometryStatus.UNCERTAIN,MetricUncertaintyStatus.UNKNOWN,("MODELLED_SYNTHETIC_CROSS_POINT_COVARIANCE",))
    @staticmethod
    def _bad(ids,code):return CrossPointCovarianceResult(tuple(ids),None,(),(),MetricGeometryStatus.INVALID,MetricUncertaintyStatus.UNKNOWN,(code,))
