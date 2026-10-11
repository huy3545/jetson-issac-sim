import hashlib,json,unittest,numpy as np
from data_source import CameraCalibration
from path_planning.cross_point_covariance import *

class CrossPointTests(unittest.TestCase):
 def setUp(self):
  self.ids=['p0','p1'];self.dim=22;std=np.r_[np.tile([.2,.3,.04,.04,.04,.04],2),[.01,.01,.02,.02],[.003,.003,.003,.002,.002,.002]];corr=np.eye(self.dim)
  for i,j,r in [(0,2,.35),(6,8,.5),(12,3,.2),(13,19,.15)]:corr[i,j]=corr[j,i]=r
  self.C=np.diag(std)@corr@np.diag(std);self.cal=CameraCalibration(100,120,50,30,np.eye(4));self.nom=(CrossPointNominal('p0',[40.25,20.5],[.5]*4,[-.04875,-.039583333,.5]),CrossPointNominal('p1',[60.75,35.25],[.5]*4,[.05375,.021875,.5]));self.prop=ExperimentalCrossPointPropagator()
 def doc(self,**u):
  shared=dict(calibration_id='cal',calibration_version='v1',transform_id='tf',transform_version='v1',shared_uncertainty_source_id='unc')
  pts=[dict(point_id=p,segment_id='s',source_id='src',camera_frame='camera',reference_frame='workpiece',timestamp=1.,clock_domain='sim',**shared) for p in self.ids]
  d=dict(schema_version='1.2-cross-point',artifact_id='a',artifact_version='v1',point_ordering=self.ids,points=pts,global_variable_ordering=list(canonical_ordering(self.ids)),global_input_covariance=self.C.tolist(),shared_provenance=shared,correlation_provenance={'status':'PROVIDED','source':'synthetic','method':'constructed PSD'},camera_frame='camera',reference_frame='workpiece',units='m',timestamp=1.,clock_domain='sim');d.update(u);return d
 def ingest(self,d):return CrossPointArtifactIngestor().ingest(json.dumps(d,allow_nan=True,separators=(',',':')).encode())
 def test_valid_artifact_digest_and_propagation_blocks(self):
  raw=json.dumps(self.doc(),separators=(',',':')).encode();a=CrossPointArtifactIngestor().ingest(raw,hashlib.sha256(raw).hexdigest());self.assertEqual(a.trust,ArtifactTrust.EVIDENCE_VALIDATED)
  r=self.prop.propagate(a.evidence,self.nom,self.cal,np.eye(3));self.assertEqual(r.global_xyz_covariance.shape,(6,6));self.assertEqual(len(r.diagonal_blocks),2);self.assertEqual(len(r.cross_blocks),1);self.assertGreater(np.linalg.norm(r.cross_blocks[0].covariance),0);self.assertFalse(r.execution_authorized);self.assertFalse(r.global_xyz_covariance.flags.writeable)
 def test_identity_ordering_and_shared_provenance_rejections(self):
  cases=[];d=self.doc();d['point_ordering']=['p0','p0'];cases.append(d);d=self.doc();d['global_variable_ordering']=[];cases.append(d);d=self.doc();d['shared_provenance']['calibration_id']='';cases.append(d);d=self.doc();d['points'][0]['transform_id']='other';cases.append(d);d=self.doc();d['points'][0]['camera_frame']='other';cases.append(d);d=self.doc();d['points'][0]['clock_domain']='other';cases.append(d);d=self.doc();d['points'][0]['timestamp']=2.;cases.append(d);d=self.doc();d['correlation_provenance']['status']='UNKNOWN';cases.append(d)
  for d in cases:self.assertEqual(self.ingest(d).trust,ArtifactTrust.INVALID)
 def test_covariance_and_digest_rejections(self):
  bad=self.C.copy();bad[0,0]=-1
  nonsymmetric=self.C.copy();nonsymmetric[0,1]+=.1
  for value in (np.eye(21),np.full((22,22),np.nan),nonsymmetric,bad):self.assertEqual(self.ingest(self.doc(global_input_covariance=value.tolist())).trust,ArtifactTrust.INVALID)
  self.assertEqual(CrossPointArtifactIngestor().ingest(b'{}','0'*64).trust,ArtifactTrust.INVALID)
 def test_global_jacobian_and_no_double_counting(self):
  ev=self.ingest(self.doc()).evidence;r=self.prop.propagate(ev,self.nom,self.cal,np.eye(3));eps=1e-6;J=np.zeros((6,22))
  for i,n in enumerate(self.nom):
   l=__import__('path_planning.full_covariance',fromlist=['compose_full_jacobian']).compose_full_jacobian(n.pixel,n.depth_samples,self.cal,np.eye(3),n.reference_xyz);J[3*i:3*i+3,6*i:6*i+6]=l[:,:6];J[3*i:3*i+3,12:16]=l[:,6:10];J[3*i:3*i+3,16:]=l[:,10:]
  q=np.r_[self.nom[0].pixel,self.nom[0].depth_samples,self.nom[1].pixel,self.nom[1].depth_samples,[100.,120.,50.,30.],[0.]*6]
  def fn(x):
   out=[]
   for i,n in enumerate(self.nom):
    u,v=x[6*i:6*i+2];a,b=u%1,v%1;w=np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b]);z=w@x[6*i+2:6*i+6];p=np.array([(u-x[14])*z/x[12],(v-x[15])*z/x[13],z]);out.extend(p+x[16:19]+np.cross(x[19:22],p))
   return np.asarray(out)
  fd=np.column_stack([(fn(q+np.eye(22)[i]*eps)-fn(q-np.eye(22)[i]*eps))/(2*eps) for i in range(22)]);np.testing.assert_allclose(J,fd,atol=2e-6,rtol=2e-6)
  # Shared extrinsics produce a nonzero off-diagonal block; they are represented once globally.
  self.assertGreater(np.linalg.norm(r.cross_blocks[0].covariance),0);self.assertTrue(np.linalg.eigvalsh(r.global_xyz_covariance).min()>=-1e-9)
 def test_monte_carlo_and_scalability_guard(self):
  ev=self.ingest(self.doc()).evidence;r=self.prop.propagate(ev,self.nom,self.cal,np.eye(3));rng=np.random.default_rng(29);samples=rng.multivariate_normal(np.zeros(self.dim),self.C,30000)
  # Linear model Monte Carlo isolates covariance composition from projection nonlinearity.
  J=np.zeros((6,22))
  for i,n in enumerate(self.nom):
   l=__import__('path_planning.full_covariance',fromlist=['compose_full_jacobian']).compose_full_jacobian(n.pixel,n.depth_samples,self.cal,np.eye(3),n.reference_xyz);J[3*i:3*i+3,6*i:6*i+6]=l[:,:6];J[3*i:3*i+3,12:16]=l[:,6:10];J[3*i:3*i+3,16:]=l[:,10:]
  empirical=np.cov(samples@J.T,rowvar=False);self.assertLess(np.linalg.norm(empirical-r.global_xyz_covariance)/np.linalg.norm(r.global_xyz_covariance),.04)
  huge=self.doc(point_ordering=[f'p{i}' for i in range(33)]);self.assertIn('POINT_LIMIT_EXCEEDED',self.ingest(huge).diagnostics)
if __name__=='__main__':unittest.main()
