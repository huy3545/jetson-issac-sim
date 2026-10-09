import unittest
import numpy as np
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, TrackingState, VisionResult, ContourType)

def result(**overrides):
    left=np.array([[0.,-.05],[1.,-.05]]); right=np.array([[0.,.05],[1.,.05]])
    fields=dict(points=left,observation_status=ObservationStatus.OBSERVED,tracking_state=TrackingState.TRACKING,confidence=1.,verified=True,orientation_source=OrientationSource.GEOMETRIC,depth_valid=np.ones(2,dtype=bool))
    l=LineVisionResult(**fields); fields['points']=right; r=LineVisionResult(**fields)
    values=dict(frame_timestamp=1.,frame_id='f1',coordinate_frame='workpiece_m',units='m',left=l,right=r)
    values.update(overrides); return VisionResult(**values)

class VisionPathAdapterTests(unittest.TestCase):
    def setUp(self): self.adapter=VisionPathAdapter()
    def test_verified_metric_observed_geometry_integrates(self):
        output=self.adapter.plan(result()); self.assertEqual(output.status,VisionPathStatus.VERIFIED); self.assertEqual(output.solver_result.overall_status.value,'CONVERGED')
    def test_predicted_expired_gapped_and_bad_frame_are_rejected(self):
        bad=LineVisionResult(points=np.array([[0.,0.],[1.,0.]]),observation_status=ObservationStatus.OCCLUDED,tracking_state=TrackingState.PREDICTING,verified=False,prediction_uncertainty_px=2)
        for vision in (result(left=bad),result(units='pixels'),result(coordinate_frame='camera'),result(frame_timestamp=None),result(time_sync_valid=False)):
            self.assertEqual(self.adapter.plan(vision).status,VisionPathStatus.REJECTED)
    def test_crossing_narrow_and_closed_are_not_verified(self):
        crossing=result(right=LineVisionResult(points=np.array([[0.,.05],[1.,-.05]]),orientation_source=OrientationSource.GEOMETRIC,depth_valid=np.ones(2,dtype=bool)))
        self.assertEqual(self.adapter.plan(crossing).status,VisionPathStatus.REJECTED)
        closed=LineVisionResult(points=np.array([[0.,0.],[1.,0.],[0.,0.]]),contour_type=ContourType.CLOSED,orientation_source=OrientationSource.GEOMETRIC,depth_valid=np.ones(3,dtype=bool))
        self.assertIn('CLOSED_CONTOUR_UNSUPPORTED',self.adapter.plan(result(left=closed)).reason_codes)
    def test_swap_and_reversal_preserve_centerline(self):
        normal=self.adapter.plan(result()).solver_result.segments[0].xy
        swapped=self.adapter.plan(result(left=result().right,right=result().left)).solver_result.segments[0].xy
        self.assertTrue(np.allclose(normal,swapped))
