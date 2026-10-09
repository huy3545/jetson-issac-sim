import unittest
import numpy as np
from path_planning.advanced_geometry import AdvancedGeometryPlanner, GeometryStatus

class AdvancedGeometryTests(unittest.TestCase):
 def setUp(self):self.p=AdvancedGeometryPlanner()
 def test_s_curve_sampling_reversal_and_variable_width(self):
  x=np.linspace(0,1,41); c=.1*np.sin(2*np.pi*x); left=np.c_[x,c-.04-.01*x]; right=np.c_[x,c+.04+.01*x]
  out=self.p.plan_partial(left,right,np.ones(len(x),bool),np.ones(len(x),bool)); self.assertEqual(out.segments[0].status,GeometryStatus.VERIFIED)
  rev=self.p.plan_partial(right[::-1],left[::-1],np.ones(len(x),bool),np.ones(len(x),bool)); self.assertEqual(rev.segments[0].status,GeometryStatus.VERIFIED)
 def test_partial_geometry_never_bridges_gap(self):
  x=np.linspace(0,1,81); left=np.c_[x,np.zeros_like(x)-.05]; right=np.c_[x,np.zeros_like(x)+.05]; valid=np.ones(len(x),bool);valid[32:48]=False
  out=self.p.plan_partial(left,right,valid,valid); self.assertEqual(len(out.segments),2);self.assertTrue(all(s.status is GeometryStatus.VERIFIED for s in out.segments));self.assertEqual(out.reason_codes,('PARTIAL_GEOMETRY',))
 def test_closed_concentric_cyclic_and_reverse_invariant(self):
  t=np.linspace(0,2*np.pi,81);a=np.c_[.4*np.cos(t),.4*np.sin(t)];b=np.c_[.5*np.cos(t),.5*np.sin(t)]
  shift=lambda p,n: np.vstack((np.roll(p[:-1],n,axis=0),np.roll(p[:-1],n,axis=0)[0]))
  one=self.p.plan_closed_concentric(a,b);two=self.p.plan_closed_concentric(shift(a,10),shift(b,20));self.assertEqual(one.status,GeometryStatus.VERIFIED);self.assertEqual(two.status,GeometryStatus.VERIFIED);self.assertAlmostEqual(one.solver_result.segments[0].width_m[0],.1)
 def test_nonconcentric_closed_is_explicitly_unsupported(self):
  t=np.linspace(0,2*np.pi,81);a=np.c_[.4*np.cos(t),.4*np.sin(t)];b=np.c_[.5*np.cos(t)+.1,.5*np.sin(t)]
  self.assertEqual(self.p.plan_closed_concentric(a,b).status,GeometryStatus.UNCERTAIN)
