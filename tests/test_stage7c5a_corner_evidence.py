"""Independent full-path evidence checks for the supported 90° synthetic fixture."""
import unittest
import numpy as np
from path_planning.corner_aware_centerline import CornerAwareTwoLineSolver
from path_planning.two_line_geometry import CurvatureStatus, TwoLineGeometryAnalyzer
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, PointDerivationMethod, TrackingState, VisionPointProvenance, VisionResult)

def distance(point,line):
    candidates=[]
    for start,end in zip(line[:-1],line[1:]):
        vector=end-start; ratio=np.clip(np.dot(point-start,vector)/np.dot(vector,vector),0.,1.)
        candidates.append(np.linalg.norm(point-(start+ratio*vector)))
    return min(candidates)
def reference_distance(point,reference): return distance(point,reference)
def inside(point,polygon):
    x,y=point; value=False
    for a,b in zip(polygon,np.vstack((polygon[1:],polygon[:1]))):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: value=not value
    return value
def self_intersects(line):
    def cross(a,b,c): return np.cross(b-a,c-a)
    for i in range(len(line)-1):
        for j in range(i+2,len(line)-1):
            if cross(line[i],line[i+1],line[j])*cross(line[i],line[i+1],line[j+1])<0 and cross(line[j],line[j+1],line[i])*cross(line[j],line[j+1],line[i+1])<0:return True
    return False

class CornerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.left=np.array([[0.,.1],[.5,.1],[.5,.6]]);self.right=np.array([[0.,-.1],[.7,-.1],[.7,.6]])
        def line(name,points):
            provenance=tuple(VisionPointProvenance(name,index,tuple(point),ObservationStatus.OBSERVED,TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,timestamp=1.,clock_domain="sim") for index,point in enumerate(points))
            return LineVisionResult(points,ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(len(points),bool),orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=provenance)
        vision=VisionResult(1.,"sim",frame_id="corner",coordinate_frame="workpiece_m",units="m",left=line("left",self.left),right=line("right",self.right))
        self.result=VisionPathAdapter(solver=CornerAwareTwoLineSolver(),require_point_provenance=True).plan(vision)

    def test_independent_complete_path_check_downgrades_90_corner(self):
        # This is the former run-midpoint candidate, independently constructed;
        # it proves why retaining the source corner would not be a medial axis.
        reference=np.array([[0.,0.],[.6,0.],[.6,.6]])
        candidate=np.vstack((np.c_[np.linspace(0,.6,141),np.zeros(141)],np.c_[np.full(140,.6),np.linspace(.6/140,.6,140)]))
        dl=np.array([distance(point,self.left) for point in candidate]);dr=np.array([distance(point,self.right) for point in candidate])
        self.assertAlmostEqual(np.max(np.abs(dl-dr)),.0414213562373095,places=12)
        self.assertTrue(all(reference_distance(point,reference)<1e-12 for point in candidate))
        self.assertEqual(self.result.status,VisionPathStatus.REJECTED);self.assertEqual(self.result.reason_codes,("CORNER_MEDIAL_AXIS_UNVERIFIED",))

    def test_rejected_corner_has_no_width_coverage_or_source_mapping_claim(self):
        self.assertEqual(self.result.status,VisionPathStatus.REJECTED)
        self.assertEqual(self.result.solver_result.segments,())
        self.assertIsNone(self.result.solver_result.diagnostics.coverage_ratio)
        # The metadata analyzer exposes unavailable values rather than output
        # indices pretending to be source identities.
        meta=TwoLineGeometryAnalyzer().analyze(self.result,self.left,self.right,coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.UNAVAILABLE)
        self.assertEqual(len(meta.source_correspondence),0);self.assertTrue(np.all(meta.source_indices==-1))

    def test_fault_conditions_are_explicit(self):
        solver=CornerAwareTwoLineSolver()
        self.assertIn("AMBIGUOUS_CORNER_CORRESPONDENCE",solver.solve(np.array([[0.,.1],[.1,.1],[.1,.6]]),np.array([[0.,-.1],[.9,-.1],[.9,.6]])).diagnostics.excluded)
        self.assertIn("TURNBACK_UNSUPPORTED",solver.solve(np.array([[0.,.1],[.5,.1],[0.,.1]]),np.array([[0.,-.1],[.5,-.1],[0.,-.1]])).diagnostics.excluded)
        rejected=solver.solve(np.array([[0.,.1],[0.,.1],[1.,.1]]),np.array([[0.,-.1],[1.,-.1]]));self.assertIn("ZERO_LENGTH_SEGMENT",rejected.diagnostics.excluded)

if __name__=="__main__":unittest.main()
