"""Independent acceptance checks for the Phase 1a planar solver."""
import json
import math
import unittest
from pathlib import Path
import numpy as np
from path_planning.planar_medial_axis import PlanarMedialAxisSolver, PlanarMedialAxisConfig, SolverStatus
from tests.phase1a_thresholds import EPSILON_SOLVER_M, HAUSDORFF_ANALYTIC_TOL_M, MAX_JUMP_FACTOR, RESAMPLE_STEP_M

DATA=json.loads((Path(__file__).parent/'fixtures'/'phase1a_open_channels.json').read_text())
def arc(case, radius):
    angle=np.linspace(case['angle_start_rad'],case['angle_end_rad'],case['samples'])
    return np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))
def point_segment(p,a,b):
    v=b-a; t=np.clip(np.dot(p-a,v)/np.dot(v,v),0,1); return np.linalg.norm(p-(a+t*v))
def distance(p,line): return min(point_segment(p,a,b) for a,b in zip(line[:-1],line[1:]))
def directed_polyline_error(points, reference):
    # Dense independent piecewise-linear sampling, rather than a vertex-set comparison.
    dense=np.vstack([a+(b-a)*np.linspace(0,1,101)[:,None] for a,b in zip(reference[:-1],reference[1:])])
    return max(min(np.linalg.norm(p-q) for q in dense) for p in points)

class PlanarMedialAxisTests(unittest.TestCase):
    def setUp(self): self.solver=PlanarMedialAxisSolver()
    def test_parallel_channel_converges_and_verifies(self):
        case=next(c for c in DATA['cases'] if c['id']=='parallel_open')
        left,right=np.array(case['left']),np.array(case['right']); result=self.solver.solve(left,right)
        self.assertEqual(result.overall_status,SolverStatus.CONVERGED); self.assertTrue(result.fully_covered)
        segment=result.segments[0]
        for p in segment.xy: self.assertLessEqual(abs(distance(p,left)-distance(p,right)),EPSILON_SOLVER_M)
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(segment.xy,axis=0),axis=1)),MAX_JUMP_FACTOR*RESAMPLE_STEP_M+EPSILON_SOLVER_M)
        self.assertLess(result.diagnostics.distance_call_count,10000)
    def test_concentric_arc_converges_and_has_analytic_centerline(self):
        case=next(c for c in DATA['cases'] if c['id']=='concentric_arc_open'); left,right=arc(case,case['r1']),arc(case,case['r2'])
        result=self.solver.solve(left,right); self.assertEqual(result.overall_status,SolverStatus.CONVERGED)
        self.assertGreaterEqual(result.segments[0].s[-1]/((case['angle_end_rad']-case['angle_start_rad'])*case['reference_radius']),.95)
        self.assertLessEqual(np.max(abs(np.linalg.norm(result.segments[0].xy,axis=1)-case['reference_radius'])),HAUSDORFF_ANALYTIC_TOL_M)
    def test_swap_and_reverse_are_invariant(self):
        left=np.array([[0.,-.05],[1.,-.05]]); right=np.array([[0.,.05],[1.,.05]])
        baseline=self.solver.solve(left,right).segments[0].xy
        for a,b in ((right,left),(left[::-1],right),(left[::-1],right[::-1])):
            other=self.solver.solve(a,b).segments[0].xy
            self.assertLessEqual(directed_polyline_error(other,baseline),HAUSDORFF_ANALYTIC_TOL_M)
    def test_invalid_inputs_do_not_raise(self):
        good=np.array([[0.,-.05],[1.,-.05]])
        for bad in (np.array([[np.nan,0],[1,0]]),np.array([[0.,0.]]),np.array([[0.,0.],[0.,0.]])):
            result=self.solver.solve(bad,good)
            self.assertEqual(result.overall_status,SolverStatus.INVALID_GEOMETRY)
        crossing=next(c for c in DATA['cases'] if c['id']=='crossing_boundaries')
        self.assertEqual(self.solver.solve(np.array(crossing['left']),np.array(crossing['right'])).overall_status,SolverStatus.INVALID_GEOMETRY)
    def test_narrow_channel_is_not_converged(self):
        result=self.solver.solve(np.array([[0.,-.001],[1.,-.001]]),np.array([[0.,.001],[1.,.001]]))
        self.assertNotEqual(result.overall_status,SolverStatus.CONVERGED)

if __name__=='__main__': unittest.main()
