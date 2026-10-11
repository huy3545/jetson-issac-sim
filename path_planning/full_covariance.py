"""Experimental full 16x16 covariance artifact and XYZ propagation."""
from dataclasses import dataclass
import hashlib, json
import numpy as np

from .uncertainty_artifact import ArtifactTrust, CrossCovariancePolicy, UncertaintyArtifactIngestor
from .xyz_uncertainty import (NumericalPropagationStatus, MetricUncertaintyStatus,
                              UncertainMetricGeometrySegment, XYZUncertaintyResult,
                              extrinsic_jacobian, pinhole_jacobian)

ORDERING = ["u","v","z00","z10","z01","z11","fx","fy","cx","cy","tx","ty","tz","rx","ry","rz"]
BLOCKS = {"pixel": (slice(0,2), (2,2)), "depth": (slice(2,6), (4,4)),
          "intrinsics": (slice(6,10), (4,4)), "extrinsics": (slice(10,16), (6,6))}

def _ro(value):
    out=np.array(value,dtype=float,copy=True);out.setflags(write=False);return out

def compose_full_jacobian(point, depth_samples, calibration, rotation, point_reference):
    """Compose the Stage 6H 3x16 Jacobian from existing Stage 6D Jacobians."""
    u,v=point;a,b=u-np.floor(u),v-np.floor(v);w=np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b])
    z=float(w@np.asarray(depth_samples,float));jc=pinhole_jacobian(u,v,z,calibration.fx,calibration.fy,calibration.cx,calibration.cy)
    J=np.zeros((3,16));J[:,:2]=rotation@jc[:,:2];J[:,2:6]=rotation@np.outer(jc[:,2],w);J[:,6:10]=rotation@jc[:,3:];J[:,10:]=extrinsic_jacobian(point_reference);return J

@dataclass(frozen=True)
class FullCovarianceEvidence:
    artifact_id: str; artifact_version: str; source_id: str; camera_frame: str; reference_frame: str
    units: str; observation_timestamp: float; clock_domain: str; calibration_evidence_reference: str
    correlation_provenance: str; full_input_covariance: np.ndarray; named_blocks: dict
    def __post_init__(self):
        object.__setattr__(self,"full_input_covariance",_ro(self.full_input_covariance))
        object.__setattr__(self,"named_blocks",{key:_ro(value) for key,value in self.named_blocks.items()})

@dataclass(frozen=True)
class FullCovarianceArtifactResult:
    trust: ArtifactTrust; diagnostics: tuple[str,...]; evidence: FullCovarianceEvidence|None
    experimental: bool=True; execution_authorized: bool=False

class FullCovarianceArtifactIngestor:
    SCHEMA_VERSION="1.1-full-covariance"
    def __init__(self,max_bytes=1_000_000): self.max_bytes=max_bytes
    def ingest(self,raw,expected_digest=None,**expected):
        if not isinstance(raw,(bytes,bytearray)): return FullCovarianceArtifactResult(ArtifactTrust.INVALID,("ARTIFACT_BYTES_REQUIRED",),None)
        if len(raw)>self.max_bytes:return FullCovarianceArtifactResult(ArtifactTrust.INVALID,("ARTIFACT_TOO_LARGE",),None)
        if expected_digest is not None and hashlib.sha256(raw).hexdigest()!=expected_digest:return FullCovarianceArtifactResult(ArtifactTrust.INVALID,("ARTIFACT_DIGEST_MISMATCH",),None)
        try:d=json.loads(bytes(raw).decode())
        except (UnicodeDecodeError,json.JSONDecodeError):return FullCovarianceArtifactResult(ArtifactTrust.INVALID,("MALFORMED_JSON",),None)
        reasons=[]
        required=("schema_version","artifact_id","artifact_version","source_id","camera_frame","reference_frame","units","observation_timestamp","clock_domain","calibration_evidence_reference","variable_ordering","covariance_blocks","full_input_covariance","cross_covariance_policy","correlation_metadata")
        reasons += ["MISSING_"+key.upper() for key in required if key not in d or d[key] in (None,"")]
        if d.get("schema_version")!=self.SCHEMA_VERSION:reasons.append("UNSUPPORTED_SCHEMA_VERSION")
        if d.get("variable_ordering")!=ORDERING:reasons.append("INVALID_VARIABLE_ORDERING")
        if d.get("units")!="m":reasons.append("UNITS_MISMATCH")
        for key,code in (("source_id","SOURCE_ID_MISMATCH"),("camera_frame","CAMERA_FRAME_MISMATCH"),("reference_frame","REFERENCE_FRAME_MISMATCH"),("observation_timestamp","TIMESTAMP_MISMATCH"),("clock_domain","CLOCK_DOMAIN_MISMATCH"),("calibration_evidence_reference","CALIBRATION_REFERENCE_MISMATCH")):
            if expected.get(key) is not None and d.get(key)!=expected[key]:reasons.append(code)
        policy=d.get("cross_covariance_policy")
        if policy!=CrossCovariancePolicy.PROVIDED_CROSS_COVARIANCE.value:reasons.append("FULL_PROPAGATION_REQUIRES_PROVIDED_CROSS_COVARIANCE")
        full=self._matrix(d.get("full_input_covariance"),(16,16),"FULL",reasons)
        blocks=d.get("covariance_blocks")
        if not isinstance(blocks,dict):reasons.append("INVALID_COVARIANCE_BLOCKS");blocks={}
        parsed={}
        for name,(sl,shape) in BLOCKS.items():
            parsed[name]=self._matrix(blocks.get(name),shape,name.upper(),reasons)
            if full is not None and parsed[name] is not None and not np.allclose(full[sl,sl],parsed[name],atol=1e-12,rtol=0.):reasons.append("NAMED_BLOCK_FULL_MATRIX_MISMATCH")
        metadata=d.get("correlation_metadata")
        if not isinstance(metadata,dict) or not metadata.get("provenance"):reasons.append("CORRELATION_PROVENANCE_MISSING")
        if reasons:return FullCovarianceArtifactResult(ArtifactTrust.INVALID,tuple(sorted(set(reasons))),None)
        evidence=FullCovarianceEvidence(d["artifact_id"],d["artifact_version"],d["source_id"],d["camera_frame"],d["reference_frame"],d["units"],d["observation_timestamp"],d["clock_domain"],d["calibration_evidence_reference"],metadata["provenance"],full,parsed)
        return FullCovarianceArtifactResult(ArtifactTrust.EVIDENCE_VALIDATED if expected_digest else ArtifactTrust.DECLARED,(),evidence)
    @staticmethod
    def _matrix(value,shape,name,reasons):
        try:m=np.asarray(value,float)
        except (TypeError,ValueError):reasons.append(f"{name}_COVARIANCE_INVALID");return None
        if m.shape!=shape:reasons.append(f"{name}_COVARIANCE_SHAPE");return None
        if not np.isfinite(m).all():reasons.append(f"{name}_COVARIANCE_NONFINITE");return None
        if not np.allclose(m,m.T,atol=1e-12,rtol=0.):reasons.append(f"{name}_COVARIANCE_NONSYMMETRIC");return None
        if np.any(np.diag(m)<0) or np.linalg.eigvalsh(m).min() < -1e-10:reasons.append(f"{name}_COVARIANCE_NOT_PSD");return None
        return m

class ExperimentalFullCovariancePropagator:
    """Uses full covariance once; never adds named blocks a second time."""
    def propagate(self,projection,vision,depth,calibration,evidence):
        metric=getattr(projection,"metric_result",None)
        if metric is None or not isinstance(evidence,FullCovarianceEvidence):return self._invalid(projection,("MISSING_FULL_COVARIANCE_EVIDENCE",))
        expected=f"{projection.calibration.evidence.calibration_id}@{projection.calibration.evidence.calibration_version}"
        checks=((evidence.source_id,metric.source_id),(evidence.camera_frame,metric.camera_frame),(evidence.reference_frame,metric.reference_frame),(evidence.units,"m"),(evidence.observation_timestamp,metric.timestamp),(evidence.clock_domain,metric.clock_domain),(evidence.calibration_evidence_reference,expected))
        if any(a!=b for a,b in checks):return self._invalid(projection,("FULL_COVARIANCE_PROVENANCE_MISMATCH",))
        segments=[]; R=metric.transform_evidence.transform_matrix[:3,:3]
        for src,line in zip(metric.segments,(vision.left,vision.right)):
            cov=[]
            for point,pref in zip(line.points,src.xyz):
                J=self.jacobian(point,depth,calibration,R,pref); out=J@evidence.full_input_covariance@J.T;out=(out+out.T)/2
                cov.append(out)
            segments.append(UncertainMetricGeometrySegment(src.name,src.xyz,np.asarray(cov),src.source_id,src.source_indices,src.status,MetricUncertaintyStatus.UNKNOWN,("MODELLED_SYNTHETIC_FULL_COVARIANCE",)))
        return XYZUncertaintyResult(tuple(segments),metric.source_id,metric.camera_frame,metric.reference_frame,"m",metric.timestamp,metric.clock_domain,metric.transform_evidence,metric.calibration_evidence,evidence,projection.calibration.trust,NumericalPropagationStatus.MODELLED_SYNTHETIC,MetricUncertaintyStatus.UNKNOWN,("MODELLED_SYNTHETIC_FULL_COVARIANCE",))
    @staticmethod
    def jacobian(point,depth,calibration,R,pref):
        u,v=point;u0,v0=np.floor(point).astype(int);u1,v1=min(u0+1,depth.shape[1]-1),min(v0+1,depth.shape[0]-1);a,b=u-u0,v-v0;w=np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b]);z=w@np.array([depth[v0,u0],depth[v0,u1],depth[v1,u0],depth[v1,u1]])
        return compose_full_jacobian(point,[depth[v0,u0],depth[v0,u1],depth[v1,u0],depth[v1,u1]],calibration,R,pref)
    @staticmethod
    def _invalid(projection,codes):
        metric=getattr(projection,"metric_result",None);return XYZUncertaintyResult((),metric.source_id if metric else None,metric.camera_frame if metric else None,metric.reference_frame if metric else None,"m",metric.timestamp if metric else None,metric.clock_domain if metric else None,metric.transform_evidence if metric else None,metric.calibration_evidence if metric else None,None,getattr(projection.calibration,"trust",None) if projection else None,NumericalPropagationStatus.INVALID,MetricUncertaintyStatus.UNKNOWN,codes)
