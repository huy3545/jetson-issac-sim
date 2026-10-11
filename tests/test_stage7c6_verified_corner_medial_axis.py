"""Independent Stage 7C-6 checks for the analytic square-corner medial axis.

These checks intentionally do not call solver-private geometry helpers.
"""
import unittest
import numpy as np

from path_planning.corner_aware_centerline import CornerAwareConfig, CornerAwareTwoLineSolver
from path_planning.two_line_geometry import CurvatureStatus, TwoLineGeometryAnalyzer
from path_planning.vision_adapter import VisionPathAdapter, VisionPathStatus
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
                              PointDerivationMethod, TrackingState,
                              VisionPointProvenance, VisionResult)


def segment_distance(point, start, end):
    vector=end-start
    fraction=np.clip(float((point-start)@vector)/float(vector@vector),0.,1.)
    return float(np.linalg.norm(point-(start+fraction*vector)))


def boundary_distance(point, line):
    return min(segment_distance(point,a,b) for a,b in zip(line[:-1],line[1:]))


def boundary_distances(points, line):
    points=np.asarray(points,float); start=line[:-1]; vector=line[1:]-start
    denominator=np.einsum("ij,ij->i",vector,vector)
    fraction=np.clip(np.einsum("nmk,mk->nm",points[:,None,:]-start[None,:,:],vector)/denominator[None,:],0.,1.)
    nearest=start[None,:,:]+fraction[:,:,None]*vector[None,:,:]
    return np.linalg.norm(points[:,None,:]-nearest,axis=2).min(axis=1)


def edge_residual(xy,left,right,subdivisions=16):
    fractions=np.linspace(0.,1.,subdivisions+1)
    points=(xy[:-1,None,:]+fractions[None,:,None]*(xy[1:,None,:]-xy[:-1,None,:])).reshape(-1,2)
    return float(np.abs(boundary_distances(points,left)-boundary_distances(points,right)).max())


def point_in_polygon(point, polygon):
    x,y=point; inside=False
    for a,b in zip(polygon,np.vstack((polygon[1:],polygon[:1]))):
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside=not inside
    return inside


def proper_intersection(a,b,c,d):
    def cross(p,q,r):
        u=q-p; v=r-p
        return float(u[0]*v[1]-u[1]*v[0])
    return cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0


class VerifiedSquareCornerTests(unittest.TestCase):
    def setUp(self):
        self.left=np.array([[0.,.1],[.5,.1],[.5,.6]])
        self.right=np.array([[0.,-.1],[.7,-.1],[.7,.6]])

    def _line(self, name, points):
        provenance=tuple(VisionPointProvenance(name,i,tuple(p),ObservationStatus.OBSERVED,
            TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,
            timestamp=1.,clock_domain="sim") for i,p in enumerate(points))
        return LineVisionResult(points,ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,
            depth_valid=np.ones(len(points),bool),orientation_source=OrientationSource.REFERENCE,
            verified=True,point_provenance=provenance)

    def _plan(self, left=None, right=None, solver=None):
        left=self.left if left is None else left; right=self.right if right is None else right
        vision=VisionResult(1.,"sim",frame_id="square-corner",coordinate_frame="workpiece_m",
            units="m",left=self._line("left",left),right=self._line("right",right))
        return VisionPathAdapter(solver=solver or CornerAwareTwoLineSolver(),require_point_provenance=True).plan(vision)

    def test_90_degree_full_path_is_independently_equidistant_contained_and_continuous(self):
        result=self._plan(); self.assertEqual(result.status,VisionPathStatus.VERIFIED)
        segment=result.solver_result.segments[0]; xy=segment.xy
        dl=np.array([boundary_distance(p,self.left) for p in xy])
        dr=np.array([boundary_distance(p,self.right) for p in xy])
        residual=np.abs(dl-dr)
        self.assertLessEqual(float(residual.max()),1e-6)
        self.assertLessEqual(float(np.sqrt(np.mean(residual**2))),1e-7)
        self.assertGreater(float(np.min(np.minimum(dl,dr))),.099999)
        polygon=np.vstack((self.left,self.right[::-1]))
        self.assertTrue(all(point_in_polygon(p,polygon) for p in xy[1:-1]))
        # Verify the continuous piecewise-linear representation, not only the
        # published samples: every output edge midpoint remains equidistant and
        # contained in the channel.
        midpoints=(xy[:-1]+xy[1:])/2
        mdl=np.array([boundary_distance(p,self.left) for p in midpoints])
        mdr=np.array([boundary_distance(p,self.right) for p in midpoints])
        self.assertLessEqual(float(np.max(np.abs(mdl-mdr))),1e-6)
        self.assertTrue(all(point_in_polygon(p,polygon) for p in midpoints))
        interpolated=np.vstack([xy[:-1]+fraction*(xy[1:]-xy[:-1]) for fraction in (.13,.37,.71,.91)])
        interpolated_residual=np.array([abs(boundary_distance(p,self.left)-boundary_distance(p,self.right)) for p in interpolated])
        self.assertLessEqual(float(interpolated_residual.max()),1e-6)
        # Production qualification samples every edge at 17 positions.
        dense=np.vstack([xy[:-1]+fraction*(xy[1:]-xy[:-1]) for fraction in np.linspace(0.,1.,17)])
        dense_residual=np.array([abs(boundary_distance(p,self.left)-boundary_distance(p,self.right)) for p in dense])
        self.assertAlmostEqual(result.solver_result.diagnostics.max_residual_m,float(dense_residual.max()),delta=1e-12)
        self.assertTrue(np.all(np.linalg.norm(np.diff(xy,axis=0),axis=1)>0))
        self.assertTrue(np.allclose(segment.s[1:],np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))))
        self.assertGreater(float(segment.s[-1]),float(np.linalg.norm(xy[-1]-xy[0])))
        self.assertFalse(any(proper_intersection(a,b,c,d) for i,(a,b) in enumerate(zip(xy[:-1],xy[1:])) for c,d in zip(xy[i+2:-1],xy[i+3:])))
        self.assertTrue(np.allclose(xy[0],[0.,0.],atol=1e-12)); self.assertTrue(np.allclose(xy[-1],[.6,.6],atol=1e-12))
        self.assertEqual(result.solver_result.diagnostics.coverage_ratio,1.)

    def test_configuration_rejects_invalid_residual_gate_inputs(self):
        for value in (0,-1,1,True,1.5,"16"):
            with self.assertRaises((TypeError,ValueError)):
                CornerAwareConfig(polyline_verification_subdivisions=value)
        valid=CornerAwareConfig(polyline_verification_subdivisions=16)
        self.assertEqual(valid.polyline_verification_subdivisions,16)
        for value in (0.,-1.,np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError): CornerAwareConfig(resample_step_m=value)
            with self.assertRaises(ValueError): CornerAwareConfig(medial_residual_tolerance_m=value)
        for field in ("structure_tolerance_m","experimental_touch_tol_m"):
            for value in (-1.,np.nan,np.inf):
                with self.assertRaises(ValueError): CornerAwareConfig(**{field:value})

    def test_interior_edge_failure_cannot_publish_converged_result(self):
        # The analytic samples have near-zero residual, while chord interiors
        # have a finite approximation residual.  A strict edge gate must see
        # that residual and reject the candidate rather than publishing it.
        loose=CornerAwareTwoLineSolver(CornerAwareConfig(medial_residual_tolerance_m=1e-5)).solve(self.left,self.right)
        self.assertTrue(loose.fully_covered)
        xy=loose.segments[0].xy
        waypoint=max(abs(boundary_distance(p,self.left)-boundary_distance(p,self.right)) for p in xy)
        interior=edge_residual(xy,self.left,self.right)
        self.assertLess(waypoint,1e-8); self.assertGreater(interior,1e-8)
        strict=CornerAwareTwoLineSolver(CornerAwareConfig(medial_residual_tolerance_m=1e-8)).solve(self.left,self.right)
        self.assertNotEqual(strict.overall_status.value,"CONVERGED")
        self.assertFalse(strict.fully_covered); self.assertEqual(strict.segments,())

    def test_90_degree_path_matches_independent_piecewise_reference(self):
        xy=self._plan().solver_result.segments[0].xy
        q=xy[:,0]-.5; y=xy[:,1]
        # In the local fixture frame: lines, a segment/vertex parabola, then a
        # vertex/segment locus.  These formulas are independent of solver code.
        join=.2*(np.sqrt(2.)-1.)
        error=[]
        for qi,yi in zip(q,y):
            if abs(qi-.1)<=1e-10 and yi>=.1-1e-10: error.append(abs(qi-.1))
            elif qi<=0: error.append(abs(yi))
            elif qi<=join+1e-12: error.append(abs(yi-qi*qi/.4))
            elif qi<=.1+1e-12: error.append(abs((yi-.1)**2-(.04-.4*qi)))
            else: error.append(abs(qi-.1))
        self.assertLessEqual(max(error),3e-7)
        # The old midpoint candidate remains demonstrably invalid.
        old=np.array([.6,0.])
        self.assertAlmostEqual(abs(boundary_distance(old,self.left)-boundary_distance(old,self.right)),.0414213562373095,places=12)

    def test_metadata_marks_only_corner_transition_unavailable(self):
        result=self._plan()
        meta=TwoLineGeometryAnalyzer().analyze(result,self.left,self.right,coordinate_frame="workpiece_m",units="m")
        self.assertEqual(meta.curvature_status,CurvatureStatus.UNAVAILABLE_AT_CORNER)
        self.assertTrue(meta.curvature_valid.any()); self.assertFalse(meta.curvature_valid.all())
        self.assertTrue(np.all(meta.source_indices==-1))
        self.assertEqual(len(meta.source_correspondence),len(meta.arc_length_m))
        self.assertTrue(all(item.derivation_method=="GEOMETRIC_SEGMENT_PROJECTION" for item in meta.source_correspondence))

    def test_curvature_is_branch_local_and_stable_across_sampling_steps(self):
        measured=[]
        for step in (.002,.005,.01):
            result=self._plan(solver=CornerAwareTwoLineSolver(CornerAwareConfig(resample_step_m=step)))
            segment=result.solver_result.segments[0]
            meta=TwoLineGeometryAnalyzer().analyze(result,self.left,self.right,coordinate_frame="workpiece_m",units="m")
            self.assertEqual(meta.curvature_status,CurvatureStatus.UNAVAILABLE_AT_CORNER)
            self.assertTrue(np.array_equal(meta.tangent_valid,meta.curvature_valid))
            self.assertTrue(np.isnan(meta.tangent_xy[~meta.tangent_valid]).all())
            # Every declared branch endpoint and its derivative stencil is
            # excluded; no gradient crosses a piecewise feature transition.
            for start,end in segment.smooth_branch_ranges:
                self.assertFalse(meta.curvature_valid[start]); self.assertFalse(meta.curvature_valid[end])
            for start,end in (segment.smooth_branch_ranges[0],segment.smooth_branch_ranges[-1]):
                index=(start+end)//2
                self.assertTrue(meta.curvature_valid[index])
                self.assertAlmostEqual(meta.curvature_m_inv[index],0.,delta=1e-9)
            parabolic_start,parabolic_end=segment.smooth_branch_ranges[1]
            index=(parabolic_start+parabolic_end)//2
            q=segment.xy[index,0]-.5
            expected=5./(1.+(q/.2)**2)**1.5
            self.assertTrue(meta.curvature_valid[index])
            self.assertAlmostEqual(meta.curvature_m_inv[index],expected,delta=.03)
            measured.append(meta.curvature_m_inv[index])
        self.assertLess(max(measured)-min(measured),.03)

    def test_reflex_turnback_and_non_square_miter_remain_fail_closed(self):
        reflex=self._plan(self.left,np.array([[0.,-.1],[.5,-.1],[.5,-.2]]))
        self.assertEqual(reflex.reason_codes,("REFLEX_ORIENTATION_UNPROVEN",))
        turnback=self._plan(np.array([[0.,.1],[.5,.1],[0.,.1]]),np.array([[0.,-.1],[.5,-.1],[0.,-.1]]))
        self.assertEqual(turnback.reason_codes,("TURNBACK_UNSUPPORTED",))
        bad=self._plan(self.left,np.array([[0.,-.1],[.69,-.1],[.7,.6]]))
        self.assertEqual(bad.reason_codes,("CORNER_MEDIAL_AXIS_UNVERIFIED",))
        unequal=self._plan(self.left,np.array([[0.,-.1],[.7,-.1],[.76,.6]]))
        self.assertEqual(unequal.reason_codes,("CORNER_MEDIAL_AXIS_UNVERIFIED",))

    def test_fatal_topology_and_input_failures_cannot_use_analytic_recovery(self):
        crossing=self._plan(np.array([[0.,.1],[1.,-.1]]),np.array([[0.,-.1],[1.,.1]]))
        self.assertEqual(crossing.reason_codes,("BOUNDARY_INTERSECTION",))
        touching=self._plan(np.array([[0.,.1],[.5,.1],[.5,.6]]),np.array([[0.,.1],[.7,.1],[.7,.6]]))
        self.assertIn("BOUNDARY_TOUCH",touching.reason_codes)
        non_finite=CornerAwareTwoLineSolver().solve(np.array([[0.,.1],[np.nan,.1],[.5,.6]]),self.right)
        self.assertIn("NON_FINITE_INPUT",non_finite.diagnostics.excluded)
        invalid_shape=CornerAwareTwoLineSolver().solve(np.array([0.,.1]),self.right)
        self.assertIn("INVALID_SHAPE",invalid_shape.diagnostics.excluded)
        scalar=CornerAwareTwoLineSolver().solve(1.,self.right)
        self.assertIn("INVALID_SHAPE",scalar.diagnostics.excluded)
        self_crossing=CornerAwareTwoLineSolver().solve(np.array([[0.,.1],[.5,.6],[0.,.6],[.5,.1]]),self.right)
        self.assertIn("SELF_INTERSECTION",self_crossing.diagnostics.excluded)

    def test_reversed_boundary_order_preserves_geometry_but_not_provenance_identity(self):
        forward=self._plan(); reverse=self._plan(right=self.right[::-1])
        self.assertEqual(reverse.status,VisionPathStatus.VERIFIED)
        self.assertTrue(np.allclose(forward.solver_result.segments[0].xy,reverse.solver_result.segments[0].xy))

    def test_rigid_transform_and_uniform_scale_preserve_qualified_geometry(self):
        baseline=self._plan().solver_result.segments[0].xy
        angle=.61; rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        offset=np.array([1.7,-.4])
        for scale in (.5,2.,5.):
            left=(self.left@rotation.T)*scale+offset; right=(self.right@rotation.T)*scale+offset
            # Build with a loose gate only to obtain the deterministic analytic
            # candidate, then make default admission agree with an independent
            # edge-distance measurement.  This does not assume 5x must pass.
            candidate=CornerAwareTwoLineSolver(CornerAwareConfig(medial_residual_tolerance_m=1.)).solve(left,right)
            self.assertTrue(candidate.fully_covered)
            xy=candidate.segments[0].xy
            interior=edge_residual(xy,left,right)
            result=self._plan(left,right)
            self.assertEqual(result.status is VisionPathStatus.VERIFIED,interior<=1e-6)
            if result.status is not VisionPathStatus.VERIFIED:
                self.assertFalse(result.solver_result.fully_covered)
                continue
            expected=(baseline@rotation.T)*scale+offset
            xy=result.solver_result.segments[0].xy
            self.assertTrue(np.allclose(xy[0],expected[0],atol=2e-12))
            self.assertTrue(np.allclose(xy[-1],expected[-1],atol=2e-12))
            residual=[abs(boundary_distance(p,left)-boundary_distance(p,right)) for p in xy]
            self.assertLessEqual(max(residual),1e-6)
            self.assertAlmostEqual(result.solver_result.diagnostics.max_residual_m,interior,delta=1e-12)


if __name__=="__main__": unittest.main()
