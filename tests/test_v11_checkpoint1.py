"""Deterministic tests for V1.1 checkpoint 1 contracts."""
import dataclasses
import unittest
import numpy as np
from config import Config
from clock import FakeClock
from data_source import TestDataSource, RGBDFrame
from machine_state import MachineState, StateSyncBuffer, SyncMode
from vision.contracts import ObservationStatus, OrientationSource, TrackingState
from vision.tracking import LineTracker
from vision.v11 import resample_arc_length, sample_depth_valid, semantic_order
from vision.vision_processor import VisionProcessor
from watchdog import ProcessHealthAdapter, WatchdogCore, WatchdogState

class CheckpointOneTests(unittest.TestCase):
    def test_phase1_immutable_and_versioned_config(self):
        cfg=Config(); self.assertEqual(cfg.config_version, Config().config_version)
        with self.assertRaises(dataclasses.FrozenInstanceError): cfg.output_dir='other'
        self.assertIn('experimental_tracking_timeout_ms', cfg.snapshot()['vision'])

    def test_phase2_fake_clock(self):
        clock=FakeClock(2.0,'sim-a'); clock.advance(.5)
        self.assertEqual((clock.now(),clock.domain_id),(2.5,'sim-a'))

    def test_phase3_sync_modes_and_domains(self):
        b=StateSyncBuffer(3); b.add(MachineState(0,0,0,0,1,'sim')); b.add(MachineState(2,4,6,8,3,'sim'))
        result=b.match(2,'sim',50); self.assertEqual(result.sync_mode,SyncMode.INTERPOLATED); self.assertEqual(result.machine_state_at_frame.y,2)
        self.assertEqual(b.match(3.02,'sim',50).sync_mode,SyncMode.EXTRAPOLATED)
        self.assertEqual(b.match(4,'sim',50).sync_mode,SyncMode.INVALID)
        self.assertEqual(b.match(2,'wall',50).reason,'CLOCK_DOMAIN_MISMATCH')

    def test_phase4_and_6_contract_depth_after_resample(self):
        frame=TestDataSource('straight').read()
        frame=RGBDFrame(frame.rgb,frame.depth,frame.calibration,timestamp=1,clock_domain='sim',rgb_timestamp=1,depth_timestamp=1)
        # Hole only at a selected resampled line coordinate, proving validity is N=resample points.
        result=VisionProcessor(Config().vision).process_frame(frame)
        self.assertEqual(len(result.left.points),100); self.assertEqual(len(result.left.depth_valid),100)
        self.assertEqual(result.left.observation_status,ObservationStatus.OBSERVED)
        self.assertEqual(result.frame_clock_domain,'sim')
        points=resample_arc_length(np.array([[0.,0.],[9.,0.]]),10)
        depth=np.ones((2,10)); depth[0,5]=np.nan
        valid,ratio,ranges=sample_depth_valid(depth,points)
        self.assertFalse(valid[5]); self.assertEqual(ranges,((0,5),(6,10))); self.assertEqual(ratio,.9)

    def test_phase5_orientation_semantics(self):
        a=np.array([[0.,0.],[10.,0.]]) ; b=np.array([[0.,10.],[10.,10.]])
        left,right,source=semantic_order(a,b); self.assertEqual(source,OrientationSource.GEOMETRIC)
        self.assertIsNotNone(left)
        ambiguous=semantic_order(a,a.copy()); self.assertEqual(ambiguous[2],OrientationSource.AMBIGUOUS)
        # Stationary machine does not participate; a supplied C-axis is stable fallback.
        x=np.array([[0.,0.],[0.,0.]]) ; y=np.array([[0.,2.],[0.,2.]])
        self.assertEqual(semantic_order(x,y,c_axis=0)[2],OrientationSource.C_AXIS)
        self.assertEqual(semantic_order(x,y,reference_left=x)[2],OrientationSource.REFERENCE)

    def test_phase7_independent_timestamp_fsm_hysteresis_and_expiry(self):
        left=LineTracker(100,20); right=LineTracker(100,20)
        self.assertEqual(left.update(True,0)[1],TrackingState.LOST) # confirm hysteresis
        self.assertEqual(left.update(True,.03)[1],TrackingState.TRACKING)
        self.assertEqual(left.update(False,.08)[1],TrackingState.PREDICTING)
        self.assertEqual(left.update(False,.14)[0],ObservationStatus.INVALID)
        self.assertEqual(right.state,TrackingState.LOST)

    def test_phase8_data_clock_watchdog_and_pause(self):
        core=WatchdogCore(100)
        self.assertEqual(core.evaluate(1.2,1.0,1.1,1.15).state,WatchdogState.CAMERA_STALE)
        self.assertEqual(core.evaluate(1.2,1.0,1.1,1.15,clock_advanced=False).state,WatchdogState.HEALTHY)
        self.assertEqual(core.evaluate(1.05,1.0,1.1,1.15).state,WatchdogState.HEALTHY)
        self.assertEqual(core.observe_frame_timestamp(2).state,WatchdogState.HEALTHY)
        self.assertEqual(core.observe_frame_timestamp(2).reason,'NON_INCREASING_FRAME_TIMESTAMP')
        clock=FakeClock(); adapter=ProcessHealthAdapter(clock,100); adapter.heartbeat(); clock.advance(.11)
        self.assertEqual(adapter.evaluate().state,WatchdogState.VISION_STALL)

    def test_sync_rejects_timestamp_and_depth_mismatch(self):
        frame=TestDataSource().read(); processor=VisionProcessor(Config().vision)
        bad=RGBDFrame(frame.rgb,frame.depth,frame.calibration,timestamp=float('nan'))
        with self.assertRaises(ValueError): processor.process_frame(bad)
        mismatch=RGBDFrame(frame.rgb,frame.depth,frame.calibration,rgb_timestamp=0,depth_timestamp=1)
        with self.assertRaises(ValueError): processor.process_frame(mismatch)

if __name__=='__main__': unittest.main()
