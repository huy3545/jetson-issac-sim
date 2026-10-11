"""Deterministic Stage 7C-4 synthetic fixture matrix; references are independent."""
import unittest
import numpy as np
from dataclasses import replace
from path_planning.two_line_geometry import CurvatureStatus, TwoLineGeometryAnalyzer
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, PointDerivationMethod,
                              TrackingState, VisionPointProvenance, VisionResult)

class TwoLineGeometryTests(unittest.TestCase):
    def _line(self,name,points):
        data=np.asarray(points,float)
        provenance=tuple(VisionPointProvenance(name,i,tuple(point),ObservationStatus.OBSERVED,TrackingState.TRACKING,
            PointDerivationMethod.DIRECT,lineage_available=True,timestamp=1.,clock_domain="sim") for i,point in enumerate(data))
        return LineVisionResult(data,ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(len(data),bool),orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=provenance)
    def _vision(self,boundary_left,boundary_right,**changes):
        values=dict(frame_timestamp=1.,frame_clock_domain="sim",frame_id="fixture",coordinate_frame="workpiece_m",units="m",left=self._line("left",boundary_left),right=self._line("right",boundary_right))
        values.update(changes);return VisionResult(**values)
    def _plan(self,left,right,**changes): return VisionPathAdapter(require_point_provenance=True).plan(self._vision(left,right,**changes))

    def test_declared_supported_straight_reference_and_metadata(self):
        left=np.array([[0.,.1],[1.,.1]]);right=np.array([[0.,-.1],[1.,-.1]])
        result=self._plan(left,right);self.assertEqual(result.status,VisionPathStatus.VERIFIED)
        meta=TwoLineGeometryAnalyzer().analyze(result,left,right,coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.AVAILABLE);self.assertTrue(meta.continuity)
        self.assertLess(np.max(np.abs(meta.curvature_m_inv)),1e-9);self.assertAlmostEqual(meta.arc_length_m[-1],.99,places=2)
        np.testing.assert_allclose(meta.tangent_xy[1:-1,1],0.,atol=1e-9)

    def test_declared_supported_gentle_curve_has_finite_independent_metadata(self):
        x=np.linspace(0,1,41);center=.1*np.sin(2*np.pi*x)
        left=np.c_[x,center+.1];right=np.c_[x,center-.1]
        result=self._plan(left,right);self.assertEqual(result.status,VisionPathStatus.VERIFIED)
        meta=TwoLineGeometryAnalyzer().analyze(result,left,right,coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.AVAILABLE);self.assertTrue(np.isfinite(meta.curvature_m_inv).all())
        # Reference centerline is the independently defined sine curve.  The
        # declared synthetic tolerance is 15 mm for this solver's 5 mm marching
        # grid and nearest-sample independent metric.
        output=result.solver_result.segments[0].xy;reference=np.c_[x,center]
        error=max(np.min(np.linalg.norm(reference-point,axis=1)) for point in output)
        self.assertLess(error,.015)

    def test_source_corner_has_no_fabricated_curvature(self):
        left=np.array([[0.,.1],[.5,.1],[.5,.6]]);right=np.array([[0.,-.1],[.7,-.1],[.7,.6]])
        result=self._plan(left,right);self.assertEqual(result.status,VisionPathStatus.VERIFIED)
        meta=TwoLineGeometryAnalyzer().analyze(result,left,right,coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.UNAVAILABLE_AT_CORNER)
        self.assertTrue(np.isnan(meta.curvature_m_inv).all());self.assertIn("CURVATURE_UNAVAILABLE_AT_SOURCE_CORNER",meta.reason_codes)

    def test_declared_rejections_have_stable_reason_codes(self):
        crossing=self._plan(np.array([[0.,.1],[1.,-.1]]),np.array([[0.,-.1],[1.,.1]]));self.assertEqual(crossing.reason_codes,("BOUNDARY_INTERSECTION",))
        duplicate=self._plan(np.array([[0.,.1],[0.,.1],[1.,.1]]),np.array([[0.,-.1],[1.,-.1]]));self.assertEqual(duplicate.reason_codes,("ZERO_LENGTH_SEGMENT",))
        zero=self._plan(np.array([[0.,0.],[1.,0.]]),np.array([[0.,0.],[1.,0.]]));self.assertEqual(zero.reason_codes,("BOUNDARY_TOUCH",))
        turnback=self._plan(np.array([[0.,.1],[.5,.1],[0.,.1]]),np.array([[0.,-.1],[.5,-.1],[0.,-.1]]));self.assertEqual(turnback.status,VisionPathStatus.REJECTED)

    def test_sampling_ordering_and_contract_faults_are_not_repaired(self):
        x=np.linspace(0,1,41);left=np.c_[x,np.full_like(x,.1)];right=np.c_[np.linspace(0,1,5),np.full(5,-.1)]
        forward=self._plan(left,right); reverse=self._plan(right[::-1],left[::-1]);self.assertEqual(forward.status,VisionPathStatus.VERIFIED);self.assertEqual(reverse.status,VisionPathStatus.VERIFIED)
        missing=self._vision(left,right,left=replace(self._line("left",left),missing_ranges=((10,12),)))
        self.assertEqual(VisionPathAdapter(require_point_provenance=True).plan(missing).reason_codes,("LEFT_GAPPED",))
        wrong_frame=self._vision(left,right,coordinate_frame="camera");self.assertEqual(VisionPathAdapter(require_point_provenance=True).plan(wrong_frame).reason_codes,("COORDINATE_FRAME_UNVERIFIED",))
        wrong_units=self._vision(left,right,units="pixels");self.assertEqual(VisionPathAdapter(require_point_provenance=True).plan(wrong_units).reason_codes,("UNITS_NOT_METRIC",))

if __name__=="__main__":unittest.main()
