import unittest
import numpy as np

from config import PathConfig
from data_source import CameraCalibration
from path_planning.path_planner import PathPlanner


def corridor(center, width):
    tangent = np.gradient(center, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return center-normal*width[:, None], center+normal*width[:, None]


def point_curve_distance(points, curve):
    output = []
    for point in points:
        a, b = curve[:-1], curve[1:]
        vector = b-a
        fraction = np.clip(((point-a)*vector).sum(axis=1)/(vector*vector).sum(axis=1), 0., 1.)
        output.append(np.linalg.norm(a+fraction[:, None]*vector-point, axis=1).min())
    return np.asarray(output)


class LegacyPathGeometryBenchmarks(unittest.TestCase):
    def setUp(self):
        self.planner = PathPlanner(PathConfig())
        self.depth = np.full((480, 640), .5)
        self.calibration = CameraCalibration(400., 400., 320., 240., np.eye(4))
        self.t = np.linspace(0., 1., 201)

    def _metric(self, pixels):
        return np.c_[(pixels[:, 0]-320.)*.5/400., (pixels[:, 1]-240.)*.5/400., np.full(len(pixels), .5)]

    def _error(self, left, right):
        truth = self._metric((left+right)/2)
        path = self.planner.plan(left, right, self.depth, self.calibration)
        return path, point_curve_distance(path, truth)

    def test_straight_large_variable_and_uneven_baselines(self):
        t = self.t
        cases = (
            (np.c_[np.full(len(t), 320.), 100+280*t], np.full(len(t), 40.), .001),
            (np.c_[320+75*np.sin(np.pi*(t-.5)), 100+280*t], np.full(len(t), 40.), .003),
            (np.c_[320+50*np.sin(np.pi*t), 100+280*t], 25+25*t, .003),
            (np.c_[320+50*np.sin(np.pi*t*t), 100+280*t*t], np.full(len(t), 40.), .003),
        )
        for center, width, maximum in cases:
            with self.subTest(maximum=maximum):
                left, right = corridor(center, width)
                _, error = self._error(left, right)
                self.assertLess(error.max(), maximum)

    def test_tight_and_s_curve_regression_expose_legacy_pairing_limit(self):
        t = self.t
        for center, width in (
            (np.c_[320+110*np.sin(2*np.pi*t), 100+280*t], np.full(len(t), 38.)),
            (np.c_[320+65*np.sin(2*np.pi*t), 100+280*t], np.full(len(t), 40.)),
        ):
            with self.subTest(center=center.shape):
                left, right = corridor(center, width)
                _, error = self._error(left, right)
                self.assertGreater(error.max(), .02)

    def test_reversed_swapped_and_intermediate_legacy_steps_are_deterministic(self):
        t = self.t
        left, right = corridor(np.c_[320+75*np.sin(np.pi*(t-.5)), 100+280*t], np.full(len(t), 40.))
        direct = self.planner.plan(left, right, self.depth, self.calibration)
        np.testing.assert_allclose(direct, self.planner.plan(left, right[::-1], self.depth, self.calibration))
        np.testing.assert_allclose(direct, self.planner.plan(right, left, self.depth, self.calibration))
        a, b = self.planner._resample(left), self.planner._resample(right)
        if np.linalg.norm(a[0]-b[-1])+np.linalg.norm(a[-1]-b[0]) < np.linalg.norm(a[0]-b[0])+np.linalg.norm(a[-1]-b[-1]):
            b = b[::-1]
        midpoint = (a+b)/2
        padded = np.pad(midpoint, ((2, 2), (0, 0)), mode="edge")
        smooth = np.column_stack([np.convolve(padded[:, axis], np.ones(5)/5, mode="valid") for axis in range(2)])
        smooth[[0, -1]] = midpoint[[0, -1]]
        expected = self._metric(smooth)
        np.testing.assert_allclose(direct, expected)

