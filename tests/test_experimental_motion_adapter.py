import dataclasses
import unittest

import numpy as np

from motion_planning.experimental_adapter import (
    ExperimentalMotionAdapter, HandoffMode, MotionHandoffConstraints,
)
from path_planning.path_qualification import (
    PathQualificationConfig, PathQualificationStatus, QualifiedPathResult, QualifiedPathSegment,
    UncertaintyStatus, qualify_surface_segments,
)
from path_planning.advanced_geometry import AdvancedGeometryPlanner
from path_planning.surface_mapping import MetricHeightmap, SourceProvenance, SurfaceMapper


def qualified(segment_id="source-1:0", source_indices=(0, 3), *, status=PathQualificationStatus.VERIFIED,
              uncertainty=UncertaintyStatus.KNOWN, frame="workpiece_m", units="m",
              timestamp=10.0, domain="simulation", reasons=()):
    xyz = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
    return QualifiedPathSegment(segment_id, 1, 0, source_indices, xyz, np.array([0., .1, .2]),
                                np.array([[1., 0., 0.]] * 3), np.zeros(3), uncertainty,
                                status, frame, units, timestamp, domain, reasons)


class ExperimentalMotionAdapterTests(unittest.TestCase):
    def result(self, *segments):
        return QualifiedPathResult("1.1", tuple(segments))

    def test_strict_accepts_complete_verified_segment_and_copies(self):
        source = qualified()
        output = ExperimentalMotionAdapter().handoff(self.result(source))
        self.assertEqual(len(output.accepted_segments), 1)
        item = output.accepted_segments[0]
        self.assertFalse(item.execution_authorized)
        self.assertFalse(item.experimental)
        self.assertFalse(item.xyz.flags.writeable)
        source.xyz[0, 0] = 99
        self.assertEqual(item.xyz[0, 0], 0)
        self.assertEqual(item.provenance, ("workpiece_m", "m", 10.0, "simulation"))

    def test_strict_rejects_unknown_and_nonverified(self):
        unknown = qualified(uncertainty=UncertaintyStatus.UNKNOWN)
        uncertain = qualified(status=PathQualificationStatus.UNCERTAIN)
        invalid = qualified(status=PathQualificationStatus.INVALID)
        output = ExperimentalMotionAdapter().handoff(self.result(unknown, uncertain, invalid))
        self.assertEqual(len(output.accepted_segments), 0)
        self.assertEqual(len(output.rejected_segments), 3)
        self.assertIn("UNCERTAINTY_NOT_VERIFIED", output.rejected_segments[0].diagnostics)
        self.assertIn("QUALIFICATION_NOT_VERIFIED", output.rejected_segments[1].diagnostics)

    def test_geometry_only_is_non_executable(self):
        output = ExperimentalMotionAdapter().handoff(
            self.result(qualified(uncertainty=UncertaintyStatus.UNKNOWN)), mode=HandoffMode.GEOMETRY_ONLY)
        self.assertEqual(len(output.accepted_segments), 1)
        item = output.accepted_segments[0]
        self.assertTrue(item.experimental)
        self.assertFalse(item.execution_authorized)
        self.assertEqual(item.uncertainty_status, UncertaintyStatus.UNKNOWN)

    def test_runs_are_isolated_and_stable(self):
        first = qualified("source-1:0", (0, 3))
        second = qualified("source-1:1", (7, 10))
        output = ExperimentalMotionAdapter().handoff(self.result(first, second))
        self.assertEqual([s.segment_id for s in output.accepted_segments], ["source-1:0", "source-1:1"])
        self.assertEqual([s.source_indices for s in output.accepted_segments], [(0, 3), (7, 10)])
        self.assertEqual([len(s.xyz) for s in output.accepted_segments], [3, 3])
        self.assertIsNot(output.accepted_segments[0].xyz, output.accepted_segments[1].xyz)

    def test_constraint_mismatches_and_blocking_diagnostics_reject(self):
        adapter = ExperimentalMotionAdapter(MotionHandoffConstraints(
            coordinate_frame="world", units="m", timestamp=11., clock_domain="robot"))
        output = adapter.handoff(self.result(qualified(reasons=("STEP_EXCEEDED",))))
        item = output.rejected_segments[0]
        for code in ("FRAME_MISMATCH", "TIMESTAMP_MISMATCH", "CLOCK_DOMAIN_MISMATCH", "BLOCKING_QUALIFICATION_DIAGNOSTIC"):
            self.assertIn(code, item.diagnostics)
        bad_units = ExperimentalMotionAdapter().handoff(self.result(qualified(units="mm")))
        self.assertIn("INVALID_PROVENANCE", bad_units.rejected_segments[0].diagnostics)

    def test_adapter_has_no_motion_or_controller_commands(self):
        output = ExperimentalMotionAdapter().handoff(self.result(qualified()))
        item = output.accepted_segments[0]
        self.assertFalse(any(hasattr(item, name) for name in ("speed", "c", "actuator", "targets")))
        self.assertFalse(any(hasattr(output, name) for name in ("execute", "send_target")))

    def test_phase2_phase3_geometry_only_input_remains_experimental(self):
        x = np.linspace(0., 1., 20)
        geometry = AdvancedGeometryPlanner().plan_partial(
            np.c_[x, np.full_like(x, -.05)], np.c_[x, np.full_like(x, .05)],
            np.ones(len(x), bool), np.ones(len(x), bool)).segments
        heightmap = MetricHeightmap(np.zeros((30, 30)), (0., -1.), (.1, .1), "workpiece_m", "m", 1.)
        mapped = SurfaceMapper().map_segments(
            geometry, heightmap, "workpiece_m", 1.,
            xy_provenance=SourceProvenance("vision", "workpiece_m", "m", 1., "simulation", "KNOWN"))
        qualified_result = qualify_surface_segments(
            mapped, coordinate_frame="workpiece_m", units="m", timestamp=1., clock_domain="simulation",
            config=PathQualificationConfig(policy="geometry_only"))
        output = ExperimentalMotionAdapter().handoff(qualified_result, mode=HandoffMode.GEOMETRY_ONLY)
        self.assertTrue(output.accepted_segments)
        self.assertTrue(all(segment.experimental and not segment.execution_authorized for segment in output.accepted_segments))
