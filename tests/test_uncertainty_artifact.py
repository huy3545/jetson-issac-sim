import hashlib, json, unittest
import numpy as np
from path_planning.uncertainty_artifact import ArtifactTrust, CrossCovariancePolicy, UncertaintyArtifactIngestor
from path_planning.xyz_uncertainty import UncertaintyValidity

class UncertaintyArtifactTests(unittest.TestCase):
 def setUp(self): self.ingestor=UncertaintyArtifactIngestor(max_bytes=20000)
 def document(self,**changes):
  d=dict(schema_version='1.0',artifact_id='u1',artifact_version='v1',source_id='frame1',camera_frame='camera',reference_frame='workpiece_m',units='m',observation_timestamp=1.,clock_domain='sim',uncertainty_source='synthetic fixture',uncertainty_method='jacobian',uncertainty_version='v1',calibration_evidence_reference='cal@v1',variable_ordering=UncertaintyArtifactIngestor.ORDERING,covariance_blocks={'pixel':[[.1,0],[0,.1]],'depth':np.eye(4).tolist(),'intrinsics':np.eye(4).tolist(),'extrinsics':np.eye(6).tolist()},cross_covariance_policy='EXPLICIT_ZERO_ASSUMPTION',correlation_metadata={'zero_cross_covariance_assumption':True,'assumption_source':'fixture'},validation_metadata={'evidence_kind':'SYNTHETIC'});d.update(changes);return d
 def ingest(self,d,**kw):
  raw=json.dumps(d,separators=(',',':'),allow_nan=True).encode();return self.ingestor.ingest(raw,**kw)
 def test_valid_synthetic_declared_and_digest_trust(self):
  d=self.document();raw=json.dumps(d,separators=(',',':')).encode()
  declared=self.ingestor.ingest(raw);validated=self.ingestor.ingest(raw,hashlib.sha256(raw).hexdigest())
  self.assertEqual(declared.trust,ArtifactTrust.DECLARED);self.assertEqual(validated.trust,ArtifactTrust.EVIDENCE_VALIDATED)
  self.assertEqual(validated.uncertainty_evidence.uncertainty_validity,UncertaintyValidity.SYNTHETIC);self.assertFalse(validated.execution_authorized)
 def test_digest_schema_ordering_and_blocks_reject(self):
  raw=b'{}';self.assertEqual(self.ingestor.ingest(raw,'0'*64).trust,ArtifactTrust.INVALID)
  for d in (self.document(schema_version='9'),self.document(variable_ordering={}),self.document(covariance_blocks={'pixel':[[1]]}),self.document(covariance_blocks={**self.document()['covariance_blocks'],'pixel':[[1,2],[0,1]]}),self.document(covariance_blocks={**self.document()['covariance_blocks'],'pixel':[[float('nan'),0],[0,1]]})):
   self.assertEqual(self.ingest(d).trust,ArtifactTrust.INVALID)
 def test_metadata_mismatch_and_missing_required_reject(self):
  for d in (self.document(source_id=''),self.document(camera_frame='other'),self.document(observation_timestamp=2.),self.document(clock_domain='other')):
   r=self.ingest(d,source_id='frame1',camera_frame='camera',observation_timestamp=1.,clock_domain='sim');self.assertEqual(r.trust,ArtifactTrust.INVALID)
 def test_cross_policy_is_preserved_and_only_explicit_zero_maps(self):
  unknown=self.ingest(self.document(cross_covariance_policy='UNKNOWN'));provided=self.ingest(self.document(cross_covariance_policy='PROVIDED_CROSS_COVARIANCE',correlation_metadata={'pixel_depth':'provided'}));zero=self.ingest(self.document())
  self.assertIsNone(unknown.uncertainty_evidence);self.assertIn('CROSS_COVARIANCE_UNKNOWN',unknown.diagnostics)
  self.assertIsNone(provided.uncertainty_evidence);self.assertIn('UNSUPPORTED_FOR_PROPAGATION',provided.diagnostics)
  self.assertIsNotNone(zero.uncertainty_evidence);self.assertEqual(zero.cross_covariance_policy,CrossCovariancePolicy.EXPLICIT_ZERO_ASSUMPTION)
 def test_bounded_malformed_immutable_and_no_dependencies(self):
  self.assertEqual(self.ingestor.ingest(b'{').trust,ArtifactTrust.INVALID);self.assertEqual(UncertaintyArtifactIngestor(max_bytes=2).ingest(b'{}{}').trust,ArtifactTrust.INVALID)
  r=self.ingest(self.document());self.assertFalse(r.uncertainty_evidence.pixel_covariance.flags.writeable)
  with self.assertRaises(ValueError):r.uncertainty_evidence.pixel_covariance[0,0]=0
  import path_planning.uncertainty_artifact as m
  with open(m.__file__) as f:s=f.read()
  for forbidden in ('from motion_planning','from safety','from machine_controller','from data_logger','from .path_planner'):self.assertNotIn(forbidden,s)
if __name__=='__main__':unittest.main()
