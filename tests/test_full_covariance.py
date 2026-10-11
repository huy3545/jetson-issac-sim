import hashlib,json,unittest,numpy as np
from data_source import CameraCalibration
from path_planning.calibration_evidence import CalibrationEvidence, CalibrationTrustedProjectionAdapter, CalibrationTrust
from path_planning.experimental_metric_projection import TransformEvidence, TransformValidity
from path_planning.full_covariance import *
from vision.contracts import LineVisionResult, ObservationStatus, OrientationSource, TrackingState, VisionResult

class FullCovarianceTests(unittest.TestCase):
 def setUp(self):
  matrix=np.eye(4);camera=CameraCalibration(100.,120.,50.,30.,matrix)
  line=lambda x:LineVisionResult(np.array([[x+.25,20.5],[x+.25,30.5]]),ObservationStatus.OBSERVED,TrackingState.TRACKING,1.,0.,np.ones(2,bool),orientation_source=OrientationSource.REFERENCE,verified=True)
  vision=VisionResult(10.,'sim',frame_id='f',coordinate_frame='camera',units='pixels',camera_to_reference=matrix,left=line(40.),right=line(60.))
  transform=TransformEvidence('camera','workpiece_m','m','i','v','e','v','fixture','fixture',9.,9.,10.,'sim','f',matrix,TransformValidity.SYNTHETIC)
  calibration_evidence=CalibrationEvidence('cal','v','cam','camera','workpiece_m','i','v','e','v','fixture','synthetic',9.,'sim','m','f',declared_trust=CalibrationTrust.SYNTHETIC)
  self.vision,self.depth,self.camera=vision,np.full((80,100),.5),camera
  self.projection=CalibrationTrustedProjectionAdapter().project(vision,self.depth,camera,transform,calibration_evidence);self.prop=ExperimentalFullCovariancePropagator()
  std=np.array([.2,.3,.04,.04,.04,.04,.01,.01,.02,.02,.003,.003,.003,.002,.002,.002]);corr=np.eye(16)
  for i,j,r in [(0,2,.35),(6,3,.25),(1,8,-.2),(7,14,.15),(5,12,.2)]:corr[i,j]=corr[j,i]=r
  self.full=np.diag(std)@corr@np.diag(std)
 def evidence(self,full=None):
  full=self.full if full is None else full
  return FullCovarianceEvidence('f','v','f','camera','workpiece_m','m',10.,'sim','cal@v','synthetic fixture',full,{n:full[s,s] for n,(s,_) in BLOCKS.items()})
 def doc(self,full=None,**changes):
  f=self.full if full is None else full;d=dict(schema_version='1.1-full-covariance',artifact_id='f',artifact_version='v',source_id='f',camera_frame='camera',reference_frame='workpiece_m',units='m',observation_timestamp=10.,clock_domain='sim',calibration_evidence_reference='cal@v',variable_ordering=ORDERING,covariance_blocks={n:f[s,s].tolist() for n,(s,_) in BLOCKS.items()},full_input_covariance=f.tolist(),cross_covariance_policy='PROVIDED_CROSS_COVARIANCE',correlation_metadata={'provenance':'synthetic fixture'});d.update(changes);return d
 def test_full_jacobian_matches_finite_difference(self):
  point=self.vision.left.points[0];pref=self.projection.metric_result.segments[0].xyz[0];J=self.prop.jacobian(point,self.depth,self.camera,np.eye(3),pref);eps=1e-6
  q=np.r_[point,[.5]*4,100.,120.,50.,30.,[0.,0.,0.,0.,0.,0.]]
  def fn(x):
   u,v=x[:2];z=np.array([.375,.125,.375,.125])@x[2:6];p=np.array([(u-x[8])*z/x[6],(v-x[9])*z/x[7],z]);return p+x[10:13]+np.cross(x[13:],p)
  fd=np.column_stack([(fn(q+np.eye(16)[i]*eps)-fn(q-np.eye(16)[i]*eps))/(2*eps) for i in range(16)])
  np.testing.assert_allclose(J,fd,atol=2e-6,rtol=2e-6)
 def test_full_cross_block_changes_result_without_double_counting(self):
  result=self.prop.propagate(self.projection,self.vision,self.depth,self.camera,self.evidence());J=self.prop.jacobian(self.vision.left.points[0],self.depth,self.camera,np.eye(3),result.segments[0].xyz[0]);expected=J@self.full@J.T
  np.testing.assert_allclose(result.segments[0].covariance_xyz[0],expected);self.assertGreater(np.max(abs(expected-J@np.diag(np.diag(self.full))@J.T)),0)
  self.assertFalse(result.segments[0].covariance_xyz.flags.writeable);self.assertFalse(result.execution_authorized)
 def test_artifact_digest_psd_blocks_and_provenance(self):
  raw=json.dumps(self.doc(),separators=(',',':')).encode();r=FullCovarianceArtifactIngestor().ingest(raw,hashlib.sha256(raw).hexdigest(),source_id='f',camera_frame='camera',reference_frame='workpiece_m',observation_timestamp=10.,clock_domain='sim',calibration_evidence_reference='cal@v')
  self.assertEqual(r.trust,ArtifactTrust.EVIDENCE_VALIDATED);self.assertFalse(r.evidence.full_input_covariance.flags.writeable)
 def test_invalid_matrix_policy_and_named_mismatch_reject(self):
  bad=self.full.copy();bad[0,0]=-1
  cases=[self.doc(full=np.eye(15)),self.doc(full=bad),self.doc(full=np.full((16,16),np.nan)),self.doc(variable_ordering=[]),self.doc(cross_covariance_policy='UNKNOWN'),self.doc(covariance_blocks={**self.doc()['covariance_blocks'],'pixel':np.eye(2).tolist()})]
  ing=FullCovarianceArtifactIngestor()
  for d in cases:self.assertEqual(ing.ingest(json.dumps(d,allow_nan=True).encode()).trust,ArtifactTrust.INVALID)
 def test_full_covariance_monte_carlo_matches_linearization(self):
  result=self.prop.propagate(self.projection,self.vision,self.depth,self.camera,self.evidence())
  predicted=result.segments[0].covariance_xyz[0];rng=np.random.default_rng(61);point=self.vision.left.points[0]
  # Perturb the full 16-vector around the same nominal state used by the Jacobian.
  q=np.r_[point,[.5]*4,100.,120.,50.,30.,[0.,0.,0.,0.,0.,0.]];samples=rng.multivariate_normal(q,self.full,30000)
  def fn(x):
   u,v=x[:2];z=np.array([.375,.125,.375,.125])@x[2:6];p=np.array([(u-x[8])*z/x[6],(v-x[9])*z/x[7],z]);return p+x[10:13]+np.cross(x[13:],p)
  empirical=np.cov(np.asarray([fn(x) for x in samples]),rowvar=False)
  self.assertLess(np.max(abs(empirical-predicted)/np.maximum(abs(predicted),1e-12)),.09)
if __name__=='__main__':unittest.main()
