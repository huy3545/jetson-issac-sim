import unittest
import cv2
import numpy as np
from config import Config
from data_source import CameraCalibration, RGBDFrame
from vision.contracts import ObservationStatus, TrackingState
from vision.runtime import ExperimentalVisionRuntime

def frame(t, lines=(True,True), shift=0, brightness=25, blur=False, mask=False, fabric=None):
    rgb=np.full((180,240,3),brightness,dtype=np.uint8)
    for present,x in zip(lines,(85+shift,155+shift)):
        if present: cv2.line(rgb,(x,20),(x,160),(0,220,0),6)
    if blur: rgb=cv2.GaussianBlur(rgb,(9,9),2)
    calibration=CameraCalibration(300,300,120,90,np.eye(4))
    return RGBDFrame(rgb,np.full((180,240),.5),calibration,timestamp=t,clock_domain='sim',frame_id=str(t),rgb_timestamp=t,depth_timestamp=t,occlusion_mask=np.ones((180,240),np.uint8) if mask else None,wheel_state={'left':'DOWN','right':'UP'},fabric_movement_uncertainty=fabric)

class VisionRuntimeTests(unittest.TestCase):
    def setUp(self): self.runtime=ExperimentalVisionRuntime(Config().vision)
    def test_roi_tracking_lighting_blur_and_reacquisition(self):
        first=self.runtime.process_frame(frame(0)); self.assertEqual(first.overall_state.value,'BOTH_OK')
        second=self.runtime.process_frame(frame(.05,shift=3,brightness=60)); self.assertEqual(second.left.tracking_state,TrackingState.TRACKING)
        self.assertIsNotNone(self.runtime.roi)
        blurred=self.runtime.process_frame(frame(.1,shift=4,blur=True)); self.assertLessEqual(blurred.left.confidence,1.)
        lost=self.runtime.process_frame(frame(.15,(False,False))); self.assertEqual(lost.left.tracking_state,TrackingState.PREDICTING)
        self.assertIsNone(self.runtime.roi)
        reacquired=self.runtime.process_frame(frame(.2,shift=8)); self.assertEqual(reacquired.overall_state.value,'BOTH_OK')
    def test_one_line_prediction_and_expiry_never_verified(self):
        self.runtime.process_frame(frame(0)); self.runtime.process_frame(frame(.05))
        one=self.runtime.process_frame(frame(.1,(True,False)))
        predicted=[line for line in (one.left,one.right) if line.tracking_state is TrackingState.PREDICTING][0]
        self.assertFalse(predicted.verified); self.assertEqual(predicted.observation_status,ObservationStatus.OCCLUDED)
        expired=self.runtime.process_frame(frame(.5,(False,False)))
        self.assertIsNone(expired.left.points); self.assertIsNone(expired.right.points)
    def test_metadata_and_timestamp_contract(self):
        result=self.runtime.process_frame(frame(0,mask=True,fabric=.3))
        self.assertEqual(result.occlusion_status,'KNOWN_MASK'); self.assertEqual(result.fabric_movement_uncertainty,.3)
        self.assertEqual(result.coordinate_frame,'camera'); self.assertIsNotNone(result.camera_to_reference)
        with self.assertRaises(ValueError): self.runtime.process_frame(frame(0))
    def test_ambiguous_single_line_is_not_verified_geometry(self):
        result=self.runtime.process_frame(frame(0,(True,False)))
        self.assertFalse(result.left.verified); self.assertFalse(result.right.verified)

if __name__=='__main__': unittest.main()
