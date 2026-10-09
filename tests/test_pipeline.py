"""Kiểm thử thuật toán, safety gate, feedback và CSV bằng unittest chuẩn."""
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from config import Config
from data_source import TestDataSource, CameraCalibration
from vision.vision_processor import VisionProcessor
from path_planning.path_planner import PathPlanner
from motion_planning.motion_planner import MotionPlanner
from safety.safety_monitor import SafetyMonitor
from controller.machine_controller import MachineController, MockMachineController
from logging_data.data_logger import DataLogger
from main import run

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.safety = SafetyMonitor(self.cfg.safety)

    def plan(self, scenario):
        frame = TestDataSource(scenario).read()
        left, right = VisionProcessor(self.cfg.vision).process(frame.rgb)
        path = PathPlanner(self.cfg.path).plan(left, right, frame.depth, frame.calibration)
        return frame, path, MotionPlanner(self.cfg.motion).plan(path)

    def test_straight_pipeline(self):
        _, path, targets = self.plan('straight')
        self.assertEqual(len(path), 100)
        self.assertLess(np.ptp(path[:, 0]), 0.002)
        np.testing.assert_allclose(path[:, 2], 0.5)
        self.assertTrue(self.safety.validate(targets).valid)
        self.assertLess(np.ptp([t['c'] for t in targets]), 0.01)

    def test_curved_pipeline_and_acceleration(self):
        _, path, targets = self.plan('curve')
        self.assertGreater(np.ptp(path[:, 0]), 0.05)
        self.assertTrue(self.safety.validate(targets).valid)
        speeds = np.array([t['speed'] for t in targets])
        self.assertEqual(speeds[0], 0)
        self.assertEqual(speeds[-1], 0)
        self.assertLess(speeds[20:-20].min(), self.cfg.motion.max_speed)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertTrue((np.abs(np.diff(speeds**2)) <= 2*self.cfg.motion.acceleration*ds + 1e-12).all())

    def test_invalid_empty_path(self):
        result = self.safety.validate([])
        self.assertFalse(result.valid)
        self.assertIn('rỗng', result.reason)

    def test_nan_inf_and_malformed_waypoints(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                self.assertFalse(self.safety.validate([dict(x=value, y=0, z=.5, c=0, speed=.01)]).valid)
        self.assertFalse(self.safety.validate([{}]).valid)
        self.assertFalse(self.safety.validate([dict(x='bad', y=0, z=.5, c=0, speed=.01)]).valid)

    def test_all_safety_limits(self):
        base = dict(x=0, y=0, z=.5, c=0, speed=.01)
        for changes, reason in [({'x': 2}, 'workspace'), ({'x': .1}, 'Khoảng cách'),
                                ({'c': 1}, 'C-axis'), ({'speed': .2}, 'Speed'),
                                ({'speed': -.1}, 'Speed')]:
            with self.subTest(changes=changes):
                result = self.safety.validate([base, dict(base, **changes)])
                self.assertFalse(result.valid)
                self.assertIn(reason, result.reason)

    def test_controller_blocks_before_sending(self):
        backend = MockMachineController()
        targets = [dict(x=0, y=0, z=.5, c=0, speed=.01), dict(x=2, y=0, z=.5, c=0, speed=.01)]
        with self.assertRaises(ValueError):
            MachineController(backend, self.safety).execute(targets)
        self.assertEqual(backend.targets, [])

    def test_feedback_monitor(self):
        class BadBackend:
            def send_target(self, target):
                return dict(x_actual=2, y_actual=0, z_actual=.5, c_actual=0)
        with self.assertRaisesRegex(ValueError, 'Feedback rejected'):
            MachineController(BadBackend(), self.safety).execute([dict(x=0, y=0, z=.5, c=0, speed=0)])

    def test_csv_end_to_end(self):
        with tempfile.TemporaryDirectory() as output:
            cfg = Config(output_dir=output)
            for scenario in ('straight', 'curve'):
                with redirect_stdout(io.StringIO()):
                    result = run(scenario, cfg)
                self.assertIsNotNone(result)
                with Path(result).open() as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    self.assertEqual(reader.fieldnames, DataLogger.fields)
                self.assertEqual(len(rows), 100)
                self.assertTrue(all(r['path_valid'] == 'True' for r in rows))
                self.assertTrue(all(r['x_actual'] == '' and r['tracking_error'] == '' for r in rows))

    def test_depth_and_extrinsic_projection(self):
        _, path, _ = self.plan('straight')
        frame = TestDataSource('straight').read()
        lines = VisionProcessor(self.cfg.vision).process(frame.rgb)
        transform = np.eye(4)
        transform[:3, 3] = [.1, .2, .3]
        camera = CameraCalibration(450, 450, 240, 180, transform)
        moved = PathPlanner(self.cfg.path).plan(*lines, frame.depth, camera)
        np.testing.assert_allclose(moved, path + [.1, .2, .3])
        for depth in (np.full(frame.depth.shape, np.nan), np.zeros_like(frame.depth)):
            with self.assertRaises(ValueError):
                PathPlanner(self.cfg.path).plan(*lines, depth, frame.calibration)

    def test_correspondence_reversed_line(self):
        frame = TestDataSource('curve').read()
        left, right = VisionProcessor(self.cfg.vision).process(frame.rgb)
        planner = PathPlanner(self.cfg.path)
        np.testing.assert_allclose(planner.plan(left, right, frame.depth, frame.calibration),
                                   planner.plan(left, right[::-1], frame.depth, frame.calibration))

    def test_angle_unwrap(self):
        angle = np.linspace(2.7, 3.6, 100)
        path = np.column_stack((.1*np.cos(angle), .1*np.sin(angle), np.full(100, .5)))
        targets = MotionPlanner(self.cfg.motion).plan(path)
        self.assertLess(np.abs(np.diff([t['c'] for t in targets])).max(), .05)

    def test_vision_missing_lines_and_error_reporting(self):
        with self.assertRaises(ValueError):
            VisionProcessor(self.cfg.vision).process(np.zeros((30, 30, 3), dtype=np.uint8))
        class BadSource:
            def read(self):
                frame = TestDataSource().read()
                frame.rgb[:] = 0
                return frame
        message = io.StringIO()
        with redirect_stdout(message):
            result = run(source=BadSource())
        self.assertIsNone(result)
        self.assertIn('Module: Vision Processing', message.getvalue())
        self.assertIn('FAILED', message.getvalue())

if __name__ == '__main__':
    unittest.main()
