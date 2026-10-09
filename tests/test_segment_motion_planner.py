import unittest
import numpy as np

from config import MotionConfig
from motion_planning.experimental_adapter import ExperimentalMotionAdapter, HandoffMode
from motion_planning.segment_motion_planner import SegmentMotionPlanner
from path_planning.path_qualification import (
    PathQualificationConfig, PathQualificationStatus, QualifiedPathResult, QualifiedPathSegment,
    UncertaintyStatus, qualify_surface_segments,
)
from path_planning.advanced_geometry import AdvancedGeometryPlanner
from path_planning.surface_mapping import MetricHeightmap, SourceProvenance, SurfaceMapper


def source(segment_id="source-1:0", indices=(0, 3), xyz=None, *, status=PathQualificationStatus.VERIFIED,
           uncertainty=UncertaintyStatus.KNOWN):
    xyz = np.array([[0., 0., .5], [.1, 0., .5], [.2, .02, .5]]) if xyz is None else np.asarray(xyz, float)
    n = len(xyz)
    return QualifiedPathSegment(segment_id, 1, indices, indices, xyz, np.arange(n, dtype=float),
                                np.ones((n, 3)), np.zeros(n), uncertainty, status,
                                "workpiece_m", "m", 10., "simulation")


class SegmentMotionPlannerTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = SegmentMotionPlanner(MotionConfig())

    def handoff(self, *segments, mode=HandoffMode.STRICT):
        result = QualifiedPathResult("1.1", tuple(segments))
        return ExperimentalMotionAdapter().handoff(result, mode=mode)

    def test_strict_segment_reuses_v1_and_preserves_contract(self):
        raw = source()
        outcome = self.wrapper.plan(self.handoff(raw))
        self.assertEqual(len(outcome.planned_segments), 1)
        plan = outcome.planned_segments[0]
        self.assertEqual((plan.segment_id, plan.source_id, plan.source_indices), ("source-1:0", 1, (0, 3)))
        self.assertEqual(plan.source_provenance, ("workpiece_m", "m", 10., "simulation"))
        self.assertTrue(plan.experimental)
        self.assertFalse(plan.execution_authorized)
        self.assertEqual(len(plan.targets), len(raw.xyz))
        np.testing.assert_allclose([[t["x"], t["y"], t["z"]] for t in plan.targets], raw.xyz)
        self.assertEqual(plan.targets[0]["speed"], 0.)
        self.assertEqual(plan.targets[-1]["speed"], 0.)

    def test_geometry_only_unknown_and_nonverified_are_rejected(self):
        geometry = self.handoff(source(uncertainty=UncertaintyStatus.UNKNOWN), mode=HandoffMode.GEOMETRY_ONLY)
        rejected = self.wrapper.plan(geometry)
        self.assertFalse(rejected.planned_segments)
        self.assertIn("STRICT_HANDOFF_REQUIRED", rejected.rejected_segments[0].diagnostics)
        unknown = self.wrapper.plan(self.handoff(source(uncertainty=UncertaintyStatus.UNKNOWN)))
        invalid = self.wrapper.plan(self.handoff(source(status=PathQualificationStatus.INVALID)))
        self.assertFalse(unknown.planned_segments)
        self.assertFalse(invalid.planned_segments)

    def test_runs_stay_isolated_without_gap_bridge(self):
        first = source("source-1:0", (0, 3))
        second = source("source-1:1", (8, 11), [[.8, .1, .5], [.9, .1, .5], [1., .12, .5]])
        result = self.wrapper.plan(self.handoff(first, second))
        self.assertEqual([p.segment_id for p in result.planned_segments], ["source-1:0", "source-1:1"])
        self.assertEqual([p.source_indices for p in result.planned_segments], [(0, 3), (8, 11)])
        self.assertEqual([len(p.targets) for p in result.planned_segments], [3, 3])
        self.assertEqual(result.planned_segments[0].targets[-1]["speed"], 0.)
        self.assertEqual(result.planned_segments[1].targets[0]["speed"], 0.)

    def test_invalid_geometry_rejects_only_its_run(self):
        good = source("source-1:0", (0, 3))
        duplicate = source("source-1:1", (5, 8), [[.5, 0., .5], [.5, 0., .5], [.6, 0., .5]])
        result = self.wrapper.plan(self.handoff(good, duplicate))
        self.assertEqual([p.segment_id for p in result.planned_segments], ["source-1:0"])
        self.assertEqual([p.segment_id for p in result.rejected_segments], ["source-1:1"])
        self.assertIn("CONSECUTIVE_DUPLICATE", result.rejected_segments[0].diagnostics)

    def test_vertical_nan_and_inf_paths_reject(self):
        paths = (
            [[0., 0., .5], [0., 0., .6], [0., 0., .7]],
            [[0., 0., .5], [np.nan, 0., .5]],
            [[0., 0., .5], [np.inf, 0., .5]],
        )
        for path in paths:
            with self.subTest(path=path):
                result = self.wrapper.plan(self.handoff(source(xyz=path)))
                self.assertFalse(result.planned_segments)
                self.assertEqual(len(result.rejected_segments), 1)

    def test_input_and_targets_are_defensively_isolated(self):
        raw = source()
        handoff = self.handoff(raw)
        result = self.wrapper.plan(handoff)
        item = result.planned_segments[0]
        raw.xyz[0, 0] = 99.
        self.assertEqual(item.targets[0]["x"], 0.)
        with self.assertRaises(TypeError):
            item.targets[0]["x"] = 4.
        self.assertFalse(any(hasattr(item, key) for key in ("execute", "send_target", "actuator", "c_axis_command")))

    def test_phase2_to_phase3_geometry_only_chain_cannot_plan_motion(self):
        x = np.linspace(0., 1., 20)
        geometry = AdvancedGeometryPlanner().plan_partial(
            np.c_[x, np.full_like(x, -.05)], np.c_[x, np.full_like(x, .05)],
            np.ones(len(x), bool), np.ones(len(x), bool)).segments
        mapped = SurfaceMapper().map_segments(
            geometry, MetricHeightmap(np.zeros((30, 30)), (0., -1.), (.1, .1), "workpiece_m", "m", 1.),
            "workpiece_m", 1., xy_provenance=SourceProvenance("vision", "workpiece_m", "m", 1., "simulation", "KNOWN"))
        qualified_result = qualify_surface_segments(
            mapped, coordinate_frame="workpiece_m", units="m", timestamp=1., clock_domain="simulation",
            config=PathQualificationConfig(policy="geometry_only"))
        handoff = ExperimentalMotionAdapter().handoff(qualified_result, mode=HandoffMode.GEOMETRY_ONLY)
        result = self.wrapper.plan(handoff)
        self.assertFalse(result.planned_segments)
        self.assertTrue(result.rejected_segments)
        self.assertTrue(all("STRICT_HANDOFF_REQUIRED" in item.diagnostics for item in result.rejected_segments))
