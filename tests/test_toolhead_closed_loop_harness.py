"""Unit tests for the six-module toolhead closed-loop harness."""
import unittest
import numpy as np

from integration.isaac_preview_adapter import CANONICAL_LEFT, CANONICAL_RIGHT
from integration.toolhead_closed_loop_harness import (
    PipelineStage,
    PipelineStatus,
    ToolheadClosedLoopHarness,
)
from vision.contracts import (
    LineVisionResult,
    ObservationStatus,
    OrientationSource,
    PointDerivationMethod,
    TrackingState,
    VisionPointProvenance,
    VisionResult,
)


def _line(name: str, points: np.ndarray) -> LineVisionResult:
    provenance = tuple(
        VisionPointProvenance(
            name,
            i,
            tuple(p),
            ObservationStatus.OBSERVED,
            TrackingState.TRACKING,
            PointDerivationMethod.DIRECT,
            lineage_available=True,
            timestamp=1.0,
            clock_domain="test_sim",
        )
        for i, p in enumerate(points)
    )
    return LineVisionResult(
        points,
        ObservationStatus.OBSERVED,
        TrackingState.TRACKING,
        0.95,
        depth_valid=np.ones(len(points), bool),
        orientation_source=OrientationSource.REFERENCE,
        verified=True,
        point_provenance=provenance,
    )


def build_canonical_vision() -> VisionResult:
    return VisionResult(
        1.0,
        "test_sim",
        frame_id="frame-square-miter-90",
        coordinate_frame="workpiece_m",
        units="m",
        left=_line("left", CANONICAL_LEFT),
        right=_line("right", CANONICAL_RIGHT),
    )


class TestToolheadClosedLoopHarness(unittest.TestCase):
    def setUp(self):
        self.harness = ToolheadClosedLoopHarness()

    def test_canonical_vision_runs_six_stages_end_to_end(self):
        vision = build_canonical_vision()
        result = self.harness.run(vision, run_id="test-run-001")

        self.assertEqual(len(result.stage_records), 6)
        self.assertTrue(result.audit_trail_complete)
        self.assertFalse(result.execution_authorized)
        self.assertEqual(result.final_status, PipelineStatus.BLOCKED)

        stages = [r.stage for r in result.stage_records]
        expected_stages = [
            PipelineStage.DATA_VISION,
            PipelineStage.PATH_PLANNING,
            PipelineStage.MOTION_PLANNING,
            PipelineStage.SAFETY_VALIDATION,
            PipelineStage.CONTROL_SIMULATION,
            PipelineStage.DATA_LOGGING,
        ]
        self.assertEqual(stages, expected_stages)

        # Vision, Path, Motion, Control, Logging are ACCEPTED; Safety is BLOCKED (Gate B)
        self.assertEqual(result.stage_records[0].status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.stage_records[1].status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.stage_records[2].status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.stage_records[3].status, PipelineStatus.BLOCKED)
        self.assertIn("STRICT_GATE_B_BLOCKED", result.stage_records[3].reason_codes)
        self.assertEqual(result.stage_records[4].status, PipelineStatus.ACCEPTED)
        self.assertEqual(result.stage_records[5].status, PipelineStatus.ACCEPTED)

        # Verify toolhead commands and simulated machine states
        self.assertIsNotNone(result.toolhead_result)
        self.assertGreater(len(result.toolhead_result.commands), 5)
        self.assertEqual(len(result.simulated_states), len(result.toolhead_result.commands))

        # Check content hashes are valid 64-char sha256 hex
        for r in result.stage_records:
            self.assertEqual(len(r.content_hash), 64)

        # Verify summary report contains essential details
        self.assertIn("JETSON SIX-MODULE PIPELINE RUN SUMMARY", result.summary_report)
        self.assertIn("STRICT_GATE_B_BLOCKED", result.summary_report)

    def test_invalid_vision_fails_closed(self):
        # Passing invalid object instead of VisionResult
        result = self.harness.run("invalid_vision")
        self.assertFalse(result.execution_authorized)
        self.assertEqual(result.stage_records[0].status, PipelineStatus.ERROR)
        self.assertEqual(result.final_status, PipelineStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
