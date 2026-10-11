"""Stage 7C-3: separate synthetic geometry proof from strict runtime admission."""
import unittest
import numpy as np
from data_source import CameraCalibration
from integration.vision_path_qualification import VisionPathQualificationHarness
from path_planning.experimental_metric_projection import TransformEvidence, TransformValidity
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, PointDerivationMethod,
    TrackingState, VisionPointProvenance, VisionResult)

def point_distance(point,line):
    values=[]
    for start,end in zip(line[:-1],line[1:]):
        vector=end-start; ratio=np.clip(np.dot(point-start,vector)/np.dot(vector,vector),0.,1.)
        values.append(np.linalg.norm(point-(start+ratio*vector)))
    return min(values)

class VisionPathQualificationTests(unittest.TestCase):
    def _line(self,name,points,*,diag=(),provenance=True,status=ObservationStatus.OBSERVED):
        data=np.asarray(points,float)
        lineage=None if not provenance else tuple(VisionPointProvenance(name,i,tuple(point),status,TrackingState.TRACKING,
            PointDerivationMethod.DIRECT,lineage_available=True,timestamp=1.,clock_domain="sim") for i,point in enumerate(data))
        return LineVisionResult(data,status,TrackingState.TRACKING,.9,depth_valid=np.ones(len(data),bool),
            orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=lineage,provenance_diagnostics=diag)
    def _metric_vision(self,boundary_left,boundary_right,**changes):
        values=dict(frame_timestamp=1.,frame_clock_domain="sim",frame_id="f",coordinate_frame="workpiece_m",units="m",
            left=self._line("left",boundary_left),right=self._line("right",boundary_right))
        values.update(changes); return VisionResult(**values)

    def test_straight_channel_uses_public_solver_and_independent_centerline_measurement(self):
        left=np.array([[0.,.1],[1.,.1]]); right=np.array([[0.,-.1],[1.,-.1]])
        result=VisionPathAdapter(require_point_provenance=True).plan(self._metric_vision(left,right))
        self.assertEqual(result.status,VisionPathStatus.VERIFIED)
        xy=result.solver_result.segments[0].xy
        residual=[abs(point_distance(point,left)-point_distance(point,right)) for point in xy]
        self.assertLess(max(residual),1e-6)
        self.assertTrue(all(min(point_distance(point,left),point_distance(point,right))>0 for point in xy))

    def test_corner_and_topology_matrix_is_explicit_not_silently_repaired(self):
        cases={
            "corner90":(np.array([[0.,.1],[.5,.1],[.5,.6]]),np.array([[0.,-.1],[.7,-.1],[.7,.6]])),
            "reflex":(np.array([[0.,.1],[.5,.1],[.5,-.4],[1.,-.4]]),np.array([[0.,-.1],[.7,-.1],[.7,-.6],[1.,-.6]])),
            "turnback":(np.array([[0.,.1],[.5,.1],[0.,.1]]),np.array([[0.,-.1],[.5,-.1],[0.,-.1]])),
            "large_curve":(np.c_[np.linspace(0,1,9),.2+.1*np.sin(np.linspace(0,np.pi/2,9))],np.c_[np.linspace(0,1,5),-.2-.1*np.sin(np.linspace(0,np.pi/2,5))]),
            "small_curve":(np.c_[np.linspace(0,.1,9),.02+.03*np.sin(np.linspace(0,np.pi/2,9))],np.c_[np.linspace(0,.1,4),-.02-.03*np.sin(np.linspace(0,np.pi/2,4))]),
        }
        adapter=VisionPathAdapter(require_point_provenance=True)
        for name,(left,right) in cases.items():
            result=adapter.plan(self._metric_vision(left,right))
            self.assertIn(result.status,(VisionPathStatus.VERIFIED,VisionPathStatus.REJECTED),name)
            if result.status is VisionPathStatus.VERIFIED:
                self.assertTrue(np.isfinite(result.solver_result.segments[0].xy).all(),name)

    def test_corner_lineage_gap_and_observation_gates_both_boundaries(self):
        left=np.array([[0.,.1],[1.,.1]]); right=np.array([[0.,-.1],[1.,-.1]])
        adapter=VisionPathAdapter(require_point_provenance=True)
        corner=self._metric_vision(left,right,left=self._line("left",left,diag=("RESAMPLING_CORNER_PRESERVATION_LIMITED",)),right=self._line("right",right,diag=("RESAMPLING_CORNER_PRESERVATION_LIMITED",)))
        self.assertEqual(adapter.plan(corner).reason_codes,("LEFT_CORNER_PRESERVATION_LIMITED","RIGHT_CORNER_PRESERVATION_LIMITED"))
        missing=self._metric_vision(left,right,left=self._line("left",left,provenance=False))
        self.assertIn("LEFT_POINT_PROVENANCE_UNAVAILABLE",adapter.plan(missing).reason_codes)
        occluded=self._metric_vision(left,right,left=self._line("left",left,status=ObservationStatus.OCCLUDED))
        self.assertIn("LEFT_NOT_OBSERVED",adapter.plan(occluded).reason_codes)

    def test_crossing_gap_and_invalid_metric_evidence_reject(self):
        left=np.array([[0.,.1],[1.,-.1]]); right=np.array([[0.,-.1],[1.,.1]])
        result=VisionPathAdapter(require_point_provenance=True).plan(self._metric_vision(left,right))
        self.assertEqual(result.status,VisionPathStatus.REJECTED)
        self.assertIn("BOUNDARY_INTERSECTION",result.reason_codes)
        gap=self._metric_vision(left,right,left=replace_line(self._line("left",left),missing_ranges=((0,1),)))
        self.assertIn("LEFT_GAPPED",VisionPathAdapter(require_point_provenance=True).plan(gap).reason_codes)

    def test_actual_runtime_sequence_stays_blocked_and_has_no_controller(self):
        matrix=np.eye(4); pixels=(np.array([[1.,1.],[2.,1.]]),np.array([[4.,1.],[5.,1.]]))
        vision=VisionResult(1.,"sim",frame_id="f",coordinate_frame="camera",units="pixels",camera_to_reference=matrix,
            left=self._line("left",pixels[0]),right=self._line("right",pixels[1]))
        evidence=TransformEvidence("camera","workpiece_m","m","i","v1","t","v1","synthetic","synthetic",0.,0.,1.,"sim","f",matrix,TransformValidity.SYNTHETIC)
        result=VisionPathQualificationHarness().evaluate(vision,np.ones((8,8)),CameraCalibration(10.,10.,0.,0.,matrix),evidence,run_id="r",correlation_id="c")
        self.assertEqual(result.bridge_status.value,"BLOCKED"); self.assertIn("UNITS_NOT_METRIC",result.reason_codes)
        self.assertIn("CALIBRATION_UNVERIFIED",result.reason_codes); self.assertFalse(result.execution_authorized)
        import integration.vision_path_qualification as module
        with open(module.__file__,encoding="utf-8") as source:
            self.assertNotIn("controller",source.read().lower())

def replace_line(line,**changes):
    from dataclasses import replace
    return replace(line,**changes)

if __name__=="__main__": unittest.main()
