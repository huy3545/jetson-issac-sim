import unittest,numpy as np
from data_source import CameraCalibration
from vision.contracts import *
from path_planning.experimental_metric_projection import *
from integration.metric_bridge import *
def fixture():
 m=np.eye(4);line=lambda x:LineVisionResult(np.array([[x,1.],[x,2.]]),ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(2,bool),orientation_source=OrientationSource.REFERENCE,verified=True)
 v=VisionResult(1.,"sim",frame_id="f",coordinate_frame="camera",units="pixels",camera_to_reference=m,left=line(1),right=line(2))
 e=TransformEvidence("camera","workpiece_m","m","i","1","t","1","synthetic","synthetic",0.,0.,1.,"sim","f",m,TransformValidity.SYNTHETIC)
 r=ExperimentalVisionMetricProjectionAdapter().project(v,np.ones((4,4)),CameraCalibration(1,1,0,0,m),e);return v,r
class T(unittest.TestCase):
 def test_preserves_uncertain_and_blocks(self):
  v,r=fixture();b=MetricGeometryBridge().convert(v,r,run_id="r",correlation_id="c");self.assertEqual(b.admission_status,AdmissionStatus.BLOCKED);self.assertEqual(b.units,"m");self.assertIn("POINT_PROVENANCE_UNAVAILABLE",b.reason_codes);self.assertEqual(b.segments[0].uncertainty_status,"UNKNOWN")
 def test_verifier_and_immutable(self):
  v,r=fixture();b=MetricGeometryBridge().convert(v,r,run_id="r",correlation_id="c");self.assertEqual(MetricBridgeVerifier().verify(b),());self.assertFalse(b.segments[0].xyz.flags.writeable)
 def test_source_identity_frame_units_and_invalid_inputs_are_not_silently_accepted(self):
  v,r=fixture();b=MetricGeometryBridge().convert(v,r,run_id="r",correlation_id="c")
  self.assertEqual((b.source_frame_id,b.timestamp,b.clock_domain,b.coordinate_frame,b.units),(v.frame_id,r.timestamp,r.clock_domain,r.reference_frame,"m"))
  bad=MetricGeometryBridge().convert(v,"not-a-contract",run_id="r",correlation_id="c")
  self.assertEqual(bad.admission_status,AdmissionStatus.REJECTED);self.assertEqual(bad.reason_codes,("INVALID_METRIC_GEOMETRY_RESULT",))
  from dataclasses import replace
  changed=replace(b,coordinate_frame="",reason_codes=tuple(code for code in b.reason_codes if code!="POINT_PROVENANCE_UNAVAILABLE"))
  reasons=MetricBridgeVerifier().verify(changed)
  self.assertIn("REFERENCE_FRAME_MISSING",reasons);self.assertIn("POINT_PROVENANCE_DISCLOSURE_MISSING",reasons)
if __name__=='__main__':unittest.main()
