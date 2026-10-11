import unittest
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from integration.isaac_preview_adapter import IsaacPreviewEvidenceAdapter
from integration.isaac_preview_contracts import IsaacPreviewArtifact
from integration.six_module_dry_run import SixModuleDryRunHarness
from simulation.isaac_preview_renderer import display_xyz
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
                              PointDerivationMethod, TrackingState, VisionPointProvenance,
                              VisionResult)


class IsaacPreviewTests(unittest.TestCase):
    def _line(self,name,points,provenance=True):
        lineage=None if not provenance else tuple(VisionPointProvenance(name,i,tuple(p),ObservationStatus.OBSERVED,TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,timestamp=2.,clock_domain="sim") for i,p in enumerate(points))
        return LineVisionResult(np.asarray(points,float),ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(len(points),bool),orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=lineage)
    def _vision(self,left=None,right=None,**changes):
        left=np.array([[0.,.1],[.5,.1],[.5,.6]]) if left is None else left
        right=np.array([[0.,-.1],[.7,-.1],[.7,.6]]) if right is None else right
        values=dict(frame_timestamp=2.,frame_clock_domain="sim",frame_id="preview-frame",coordinate_frame="workpiece_m",units="m",left=self._line("left",left),right=self._line("right",right))
        values.update(changes); return VisionResult(**values)
    def _build(self,vision=None):
        vision=vision or self._vision(); trace=SixModuleDryRunHarness().run(vision,run_id="preview-run",correlation_id="preview-corr")
        return IsaacPreviewEvidenceAdapter().build(vision,trace),trace

    def test_square_fixture_uses_solver_centerline_and_is_read_only(self):
        artifact,trace=self._build()
        self.assertTrue(artifact.preview_authorized); self.assertFalse(artifact.execution_authorized)
        self.assertEqual(artifact.geometry_status,"SYNTHETIC_GEOMETRY_VERIFIED")
        self.assertEqual(trace.records[1].status.value,"BLOCKED")
        self.assertGreater(len(artifact.centerline_xy),2)
        self.assertEqual(artifact.left_boundary_xy,self._build()[0].left_boundary_xy)
        self.assertEqual(display_xyz(((1.,2.),),artifact.display_transform),((1.,2.,0.),))
        with self.assertRaises(TypeError): artifact.left_boundary_xy[0][0]=8

    def test_json_round_trip_and_invalid_numeric_are_rejected(self):
        artifact,_=self._build(); restored=IsaacPreviewArtifact.from_json(artifact.to_json())
        self.assertEqual(restored.centerline_xy,artifact.centerline_xy)
        raw=json.loads(artifact.to_json()); raw["left_boundary_xy"][0][0]=float("nan")
        with self.assertRaises(ValueError): IsaacPreviewArtifact.from_json(json.dumps(raw))

    def test_rejected_geometry_has_no_verified_centerline(self):
        reflex=self._vision(right=np.array([[0.,-.1],[.5,-.1],[.5,-.2]])); artifact,_=self._build(reflex)
        self.assertFalse(artifact.preview_authorized); self.assertIsNone(artifact.centerline_xy)
        self.assertIn("REFLEX_ORIENTATION_UNPROVEN",artifact.reason_codes)
        crossing=self._vision(left=np.array([[0.,.1],[1.,-.1]]),right=np.array([[0.,-.1],[1.,.1]])); bad,_=self._build(crossing)
        self.assertFalse(bad.preview_authorized); self.assertIn("BOUNDARY_INTERSECTION",bad.reason_codes)

    def test_identity_and_observation_failures_are_conservative(self):
        vision=self._vision(); _,trace=self._build(vision)
        mismatched=replace(vision,frame_id="other-frame")
        artifact=IsaacPreviewEvidenceAdapter().build(mismatched,trace)
        self.assertFalse(artifact.preview_authorized); self.assertIn("TRACE_SOURCE_IDENTITY_MISMATCH",artifact.reason_codes)
        invalid=replace(vision,left=self._line("left",np.array([[0.,.1],[.5,.1],[.5,.6]]),False))
        artifact,_=self._build(invalid); self.assertFalse(artifact.preview_authorized)

    def test_units_and_renderer_runtime_are_not_silently_supported(self):
        vision=self._vision(units="pixels"); trace=SixModuleDryRunHarness().run(vision,run_id="p",correlation_id="p")
        with self.assertRaises(ValueError): IsaacPreviewEvidenceAdapter().build(vision,trace)
        self.assertNotIn("MachineController",Path("simulation/isaac_preview_renderer.py").read_text())


if __name__ == "__main__": unittest.main()
