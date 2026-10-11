"""Stage 7C-5 quantified synthetic support; strict runtime admission is untouched."""
import unittest
import numpy as np
from dataclasses import replace
from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.two_line_geometry import CurvatureStatus, TwoLineGeometryAnalyzer
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, PointDerivationMethod,
                              TrackingState, VisionPointProvenance, VisionResult)

class CornerAwareTests(unittest.TestCase):
    def _line(self,name,points,*,provenance=True):
        data=np.asarray(points,float); lineage=None if not provenance else tuple(VisionPointProvenance(name,index,tuple(point),ObservationStatus.OBSERVED,TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,timestamp=1.,clock_domain="sim") for index,point in enumerate(data))
        return LineVisionResult(data,ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(len(data),bool),orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=lineage)
    def _plan(self,boundary_left,boundary_right,*,solver=None,**changes):
        values=dict(frame_timestamp=1.,frame_clock_domain="sim",frame_id="corner",coordinate_frame="workpiece_m",units="m",left=self._line("left",boundary_left),right=self._line("right",boundary_right));values.update(changes)
        return VisionPathAdapter(solver=solver or CornerAwareTwoLineSolver(),require_point_provenance=True).plan(VisionResult(**values))

    def test_sharp_90_corner_is_rejected_without_medial_axis_evidence(self):
        left=np.array([[0.,.1],[.5,.1],[.5,.6]]);right=np.array([[0.,-.1],[.7,-.1],[.7,.6]])
        result=self._plan(left,right);self.assertEqual(result.status,VisionPathStatus.REJECTED)
        self.assertEqual(result.reason_codes,("CORNER_MEDIAL_AXIS_UNVERIFIED",))

    def test_straight_s_curve_unequal_count_and_uneven_sampling_remain_supported(self):
        straight=self._plan(np.array([[0.,.1],[1.,.1]]),np.array([[0.,-.1],[1.,-.1]]));self.assertEqual(straight.status,VisionPathStatus.VERIFIED)
        x=np.linspace(0,1,41);center=.1*np.sin(2*np.pi*x)
        uneven=np.r_[0.,np.linspace(.02,.8,7),1.]
        curve=self._plan(np.c_[x,center+.1],np.c_[uneven,np.interp(uneven,x,center)-.1]);self.assertEqual(curve.status,VisionPathStatus.VERIFIED)
        meta=TwoLineGeometryAnalyzer().analyze(curve,np.c_[x,center+.1],np.c_[x,center-.1],coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.AVAILABLE);self.assertTrue(meta.curvature_valid.all())

    def test_reflex_turnback_and_ambiguous_correspondence_reject_stably(self):
        reflex=self._plan(np.array([[0.,.1],[.5,.1],[.5,.6]]),np.array([[0.,-.1],[.7,-.1],[.7,-.6]]));self.assertEqual(reflex.reason_codes,("REFLEX_ORIENTATION_UNPROVEN",))
        turnback=self._plan(np.array([[0.,.1],[.5,.1],[0.,.1]]),np.array([[0.,-.1],[.5,-.1],[0.,-.1]]));self.assertEqual(turnback.reason_codes,("TURNBACK_UNSUPPORTED",))
        ambiguous=self._plan(np.array([[0.,.1],[.1,.1],[.1,.6]]),np.array([[0.,-.1],[.9,-.1],[.9,.6]]));self.assertEqual(ambiguous.reason_codes,("AMBIGUOUS_CORNER_CORRESPONDENCE",))

    def test_faults_keep_existing_rejections_and_strict_contract(self):
        crossing=self._plan(np.array([[0.,.1],[1.,-.1]]),np.array([[0.,-.1],[1.,.1]]));self.assertEqual(crossing.reason_codes,("BOUNDARY_INTERSECTION",))
        zero=self._plan(np.array([[0.,0.],[1.,0.]]),np.array([[0.,0.],[1.,0.]]));self.assertEqual(zero.reason_codes,("BOUNDARY_TOUCH",))
        left=np.array([[0.,.1],[1.,.1]]);right=np.array([[0.,-.1],[1.,-.1]])
        missing=self._plan(left,right,left=self._line("left",left,provenance=False));self.assertEqual(missing.reason_codes,("LEFT_POINT_PROVENANCE_UNAVAILABLE",))
        wrong=self._plan(left,right,units="pixels");self.assertEqual(wrong.reason_codes,("UNITS_NOT_METRIC",))

if __name__=="__main__":unittest.main()
