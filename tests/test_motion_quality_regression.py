"""Deterministic numerical regression baselines from Motion Stage 4 audit."""
import unittest

import numpy as np

from config import MotionConfig
from motion_planning.experimental_adapter import ExperimentalMotionAdapter
from motion_planning.motion_planner import MotionPlanner
from motion_planning.segment_motion_planner import SegmentMotionPlanner
from path_planning.path_qualification import (
    PathQualificationStatus, QualifiedPathResult, QualifiedPathSegment, UncertaintyStatus,
)


def target_values(targets):
    return np.array([[target[key] for key in ("x", "y", "z", "c", "speed")] for target in targets])


def local_circle_curvature(points):
    """Independent three-point circumcircle estimate for planar samples."""
    a, b, c = points[:-2, :2], points[1:-1, :2], points[2:, :2]
    ab, bc, ca = np.linalg.norm(b-a, axis=1), np.linalg.norm(c-b, axis=1), np.linalg.norm(a-c, axis=1)
    twice_area = np.abs((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))
    return 2*twice_area/(ab*bc*ca)


def planner_style_curvature(path):
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0., np.cumsum(ds)]
    tangent = np.gradient(path, s, axis=0)
    unit = tangent / np.linalg.norm(tangent, axis=1)[:, None]
    return np.linalg.norm(np.gradient(unit, s, axis=0), axis=1)


def qualified(segment_id, indices, xyz):
    n = len(xyz)
    return QualifiedPathSegment(
        segment_id, 7, 0, indices, np.asarray(xyz, float), np.arange(n, dtype=float),
        np.ones((n, 3)), np.zeros(n), UncertaintyStatus.KNOWN,
        PathQualificationStatus.VERIFIED, "workpiece_m", "m", 5., "simulation")


class MotionQualityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MotionConfig()
        self.planner = MotionPlanner(self.cfg)
        self.n = 101
        self.t = np.linspace(0., np.pi/2, self.n)

    def plan(self, path):
        return target_values(self.planner.plan(np.asarray(path, float)))

    def assert_motion_limits(self, path, values):
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[:, 4] >= 0).all())
        self.assertLessEqual(values[:, 4].max(), self.cfg.max_speed + 1e-12)
        self.assertTrue((np.abs(np.diff(values[:, 4]**2)) <= 2*self.cfg.acceleration*ds + 1e-12).all())
        self.assertEqual(values[0, 4], 0.)
        self.assertEqual(values[-1, 4], 0.)

    def test_straight_arc_length_tangent_c_axis_and_speed(self):
        path = np.c_[np.linspace(0., .5, self.n), np.zeros(self.n), np.full(self.n, .5)]
        values = self.plan(path)
        self.assertAlmostEqual(np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), .5, places=12)
        np.testing.assert_allclose(values[:, 3], 0., atol=1e-12)
        self.assert_motion_limits(path, values)

    def test_circular_arc_analytic_tangent_and_lateral_cap(self):
        radius = .05
        path = np.c_[radius*np.cos(self.t), radius*np.sin(self.t), np.full(self.n, .5)]
        values = self.plan(path)
        expected_c = self.t + np.pi/2
        np.testing.assert_allclose(values[1:-1, 3], expected_c[1:-1], atol=2e-4)
        # The discrete curvature is 19.99938 rather than analytic 20 1/m.
        # Allow only its measured floating-point/finite-difference excess.
        self.assertTrue((values[:, 4]**2 / radius <= self.cfg.lateral_acceleration + 2e-6).all())
        self.assertLessEqual(values[:, 4].max(), np.sqrt(self.cfg.lateral_acceleration/radius) + 1e-10)
        self.assert_motion_limits(path, values)

    def test_circular_curvature_interior_and_endpoint_baseline(self):
        radius = 1.
        path = np.c_[radius*np.cos(self.t), radius*np.sin(self.t), np.full(self.n, .5)]
        independent = local_circle_curvature(path)
        self.assertTrue(np.allclose(independent, 1/radius, rtol=2e-4, atol=2e-4))
        measured = planner_style_curvature(path)
        self.assertAlmostEqual(measured[0], .5/radius, delta=2e-4)
        self.assertAlmostEqual(measured[-1], .5/radius, delta=2e-4)
        self.assertAlmostEqual(measured[self.n//2], 1/radius, delta=2e-4)

    def test_nonuniform_sampling_and_z_change_remain_finite(self):
        u = np.linspace(0., 1., self.n)**2
        uneven = np.c_[.25*np.cos(np.pi/2*u), .25*np.sin(np.pi/2*u), np.full(self.n, .5)]
        z_change = np.c_[.2*np.cos(self.t), .2*np.sin(self.t), .4+.1*self.t/(np.pi/2)]
        for path in (uneven, z_change):
            with self.subTest(path=path.shape):
                values = self.plan(path)
                self.assert_motion_limits(path, values)
                self.assertTrue(np.isfinite(np.diff(values[:, 3])).all())

    def test_s_curve_unwrap_and_acceleration(self):
        x = np.linspace(0., .5, self.n)
        path = np.c_[x, .05*np.sin(4*np.pi*x/.5), np.full(self.n, .5)]
        values = self.plan(path)
        self.assertLess(np.abs(np.diff(values[:, 3])).max(), .2)
        self.assert_motion_limits(path, values)

    def test_fixed_seed_noise_curvature_conditioning_baseline(self):
        ideal = np.c_[np.cos(self.t), np.sin(self.t), np.full(self.n, .5)]
        rng = np.random.default_rng(4)
        noisy = ideal + rng.normal(0., 3e-4, ideal.shape)
        noisy[:, 2] = .5
        ideal_error = np.abs(local_circle_curvature(ideal)-1.).max()
        noisy_error = np.abs(local_circle_curvature(noisy)-1.).max()
        self.assertLess(ideal_error, 2e-4)
        self.assertGreater(noisy_error, .5)  # Regression baseline: 0.3 mm noise is significant.
        self.assert_motion_limits(noisy, self.plan(noisy))

    def test_short_segment_and_degenerate_rejection(self):
        short = np.array([[0., 0., .5], [.02, .01, .5]])
        values = self.plan(short)
        self.assertEqual(list(values[:, 4]), [0., 0.])
        bad_paths = (
            [[0., 0., .5], [0., 0., .6], [0., 0., .7]],
            [[0., 0., .5], [1e-11, 0., .6], [2e-11, 0., .7]],
            [[0., 0., .5], [0., 0., .5], [.1, 0., .5]],
        )
        for path in bad_paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.plan(path)

    def test_segment_runs_keep_boundaries_provenance_and_gap(self):
        first = qualified("run-0", (0, 3), [[0., 0., .5], [.1, 0., .5], [.2, 0., .5]])
        second = qualified("run-1", (8, 11), [[.8, .1, .5], [.9, .1, .5], [1., .1, .5]])
        handoff = ExperimentalMotionAdapter().handoff(QualifiedPathResult("1.1", (first, second)))
        result = SegmentMotionPlanner(self.cfg).plan(handoff)
        self.assertEqual([segment.source_indices for segment in result.planned_segments], [(0, 3), (8, 11)])
        self.assertEqual([len(segment.targets) for segment in result.planned_segments], [3, 3])
        self.assertTrue(all(s.targets[0]["speed"] == 0. and s.targets[-1]["speed"] == 0. for s in result.planned_segments))
        self.assertTrue(all(s.source_provenance == ("workpiece_m", "m", 5., "simulation") for s in result.planned_segments))
        self.assertTrue(all(not s.execution_authorized for s in result.planned_segments))
