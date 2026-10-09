"""Immutable V1 compatibility regression fixture."""
import hashlib
import json
import platform
import unittest
from pathlib import Path

import cv2
import numpy as np

from config import Config
from data_source import TestDataSource
from vision.vision_processor import VisionProcessor
from path_planning.path_planner import PathPlanner
from motion_planning.motion_planner import MotionPlanner
from safety.safety_monitor import SafetyMonitor

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "v1_golden_straight.json"
EXPECTED_HASH = (ROOT / "fixtures" / "v1_golden_straight.sha256").read_text().strip()


class GoldenRegressionTests(unittest.TestCase):
    def test_fixture_integrity(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), EXPECTED_HASH)

    def test_environment_metadata(self):
        fixture = json.loads(FIXTURE.read_text())
        environment = fixture["environment"]
        actual = {"numpy": np.__version__, "opencv": cv2.__version__}
        mismatches = {key: (environment[key], value) for key, value in actual.items()
                      if environment[key] != value}
        if mismatches:
            self.skipTest("Golden numerical environment mismatch: " + repr(mismatches))
        self.assertEqual(environment["python"], platform.python_version())

    def test_straight_compatibility_targets(self):
        fixture = json.loads(FIXTURE.read_text())
        cfg = Config()
        frame = TestDataSource("straight").read()
        left, right = VisionProcessor(cfg.vision).process(frame.rgb)
        path = PathPlanner(cfg.path).plan(left, right, frame.depth, frame.calibration)
        targets = MotionPlanner(cfg.motion).plan(path)
        self.assertTrue(SafetyMonitor(cfg.safety).validate(targets).valid)
        self.assertEqual(len(targets), len(fixture["targets"]))
        for index, (actual, expected) in enumerate(zip(targets, fixture["targets"])):
            self.assertEqual(index, expected["target_index"])
            for field in ("x", "y", "z", "c", "speed"):
                self.assertAlmostEqual(actual[field], expected[field], places=10,
                                       msg=f"target={index}, field={field}")
