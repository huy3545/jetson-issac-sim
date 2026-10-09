import unittest

import numpy as np

from data_source import CameraCalibration
from path_planning.experimental_geometry_entrypoint import (
    BenchmarkStatus,
    ExperimentalGeometryEntrypoint,
    LegacyBenchmarkContext,
)
from path_planning.legacy_geometry_gate import (
    GeometryEvidence,
    GeometryGateStatus,
    LegacyGeometryGateConfig,
)


class _SpyPlanner:
    def __init__(self, output=None, error=None):
        self.calls = 0
        self.output = np.asarray(output if output is not None else [[1., 2., 3.], [4., 5., 6.]])
        self.error = error

    def plan(self, *args):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


class ExperimentalGeometryEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.left = np.array([[0., 0.], [1., 0.], [2., 0.]])
        self.right = np.array([[0., 1.], [1., 1.], [2., 1.]])
        self.evidence = GeometryEvidence(
            coordinate_frame="world", units="m", timestamp=10.0, clock_domain="sim",
            source_id="synthetic-channel", topology_verified=True,
            correspondence_verified=True, reference_centerline=np.array([[0., .5], [2., .5]]),
        )
        self.gate_config = LegacyGeometryGateConfig(
            max_reference_deviation_m=1e-8, max_reference_rms_m=1e-8,
        )
        self.context = LegacyBenchmarkContext(
            left_boundary_pixels=np.array([[20., 10.], [30., 20.], [40., 30.]]),
            right_boundary_pixels=np.array([[20., 20.], [30., 30.], [40., 40.]]),
            depth=np.full((64, 64), .5),
            calibration=CameraCalibration(100., 100., 32., 32., np.eye(4)),
            source_id="synthetic-channel", timestamp=10.0, clock_domain="sim",
            synthetic=True, mapping_provenance="declared synthetic pixel-to-metric fixture",
            mapping_verified=True,
        )

    def _entrypoint(self, spy=None):
        return ExperimentalGeometryEntrypoint(self.gate_config, path_planner=spy or _SpyPlanner())

    def test_accept_calls_legacy_once_and_never_authorizes_execution(self):
        spy = _SpyPlanner()
        result = self._entrypoint(spy).run(self.left, self.right, self.evidence, self.context)
        self.assertEqual(result.gate_result.status, GeometryGateStatus.ACCEPT)
        self.assertEqual(result.benchmark_status, BenchmarkStatus.SUCCESS)
        self.assertEqual(spy.calls, 1)
        self.assertTrue(result.experimental)
        self.assertFalse(result.execution_authorized)
        self.assertFalse(result.output_path.flags.writeable)
        np.testing.assert_allclose(result.output_path, spy.output)

    def test_reject_and_uncertain_never_invoke_legacy_planner(self):
        reject_spy = _SpyPlanner()
        crossing = np.array([[0., 1.], [1., -1.], [2., 1.]])
        rejected = self._entrypoint(reject_spy).run(self.left, crossing, self.evidence, self.context)
        self.assertEqual(rejected.gate_result.status, GeometryGateStatus.REJECT)
        self.assertEqual(rejected.benchmark_status, BenchmarkStatus.NOT_RUN)
        self.assertEqual(reject_spy.calls, 0)

        uncertain_spy = _SpyPlanner()
        uncertain_evidence = GeometryEvidence(reference_centerline=self.evidence.reference_centerline)
        uncertain = self._entrypoint(uncertain_spy).run(self.left, self.right, uncertain_evidence, self.context)
        self.assertEqual(uncertain.gate_result.status, GeometryGateStatus.UNCERTAIN)
        self.assertEqual(uncertain.benchmark_status, BenchmarkStatus.NOT_RUN)
        self.assertEqual(uncertain_spy.calls, 0)

    def test_invalid_or_missing_context_stops_after_accept(self):
        spy = _SpyPlanner()
        result = self._entrypoint(spy).run(self.left, self.right, self.evidence)
        self.assertEqual(result.gate_result.status, GeometryGateStatus.ACCEPT)
        self.assertEqual(result.benchmark_status, BenchmarkStatus.CONTEXT_INVALID)
        self.assertIn("MISSING_BENCHMARK_CONTEXT", result.reason_codes)
        self.assertEqual(spy.calls, 0)

    def test_context_requires_explicit_mapping_and_matching_metadata(self):
        spy = _SpyPlanner()
        incomplete = LegacyBenchmarkContext(
            **{**self.context.__dict__, "mapping_verified": False, "source_id": "other"}
        )
        result = self._entrypoint(spy).run(self.left, self.right, self.evidence, incomplete)
        self.assertEqual(result.benchmark_status, BenchmarkStatus.CONTEXT_INVALID)
        self.assertIn("MISSING_BOUNDARY_MAPPING_PROVENANCE", result.reason_codes)
        self.assertIn("BENCHMARK_CONTEXT_METADATA_MISMATCH", result.reason_codes)
        self.assertEqual(spy.calls, 0)

    def test_snapshots_and_output_do_not_share_mutable_arrays(self):
        spy = _SpyPlanner()
        result = self._entrypoint(spy).run(self.left, self.right, self.evidence, self.context)
        self.evidence.reference_centerline[0, 0] = 99.
        spy.output[0, 0] = 99.
        self.assertEqual(result.evidence.reference_centerline[0, 0], 0.)
        self.assertEqual(result.output_path[0, 0], 1.)
        with self.assertRaises(ValueError):
            result.output_path[0, 0] = 0.

    def test_benchmark_failure_is_separate_from_gate_acceptance(self):
        spy = _SpyPlanner(error=ValueError("depth failed"))
        result = self._entrypoint(spy).run(self.left, self.right, self.evidence, self.context)
        self.assertEqual(result.gate_result.status, GeometryGateStatus.ACCEPT)
        self.assertEqual(result.benchmark_status, BenchmarkStatus.BENCHMARK_FAILED)
        self.assertEqual(result.reason_codes, ("LEGACY_PATH_BENCHMARK_FAILED", "ValueError"))
        self.assertEqual(spy.calls, 1)

    def test_actual_legacy_benchmark_returns_immutable_path(self):
        result = ExperimentalGeometryEntrypoint(self.gate_config).run(
            self.left, self.right, self.evidence, self.context,
        )
        self.assertEqual(result.benchmark_status, BenchmarkStatus.SUCCESS)
        self.assertEqual(result.output_path.shape, (100, 3))
        self.assertTrue(np.isfinite(result.output_path).all())
        self.assertFalse(result.output_path.flags.writeable)


if __name__ == "__main__":
    unittest.main()
