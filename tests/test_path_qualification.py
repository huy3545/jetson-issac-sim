import unittest,numpy as np
from path_planning.surface_mapping import SurfaceSegment,SurfaceStatus,SurfaceMappingResult
from path_planning.path_qualification import *
class Q(unittest.TestCase):
 def s(self,x):return SurfaceMappingResult((SurfaceSegment(2,np.array(x,float),SurfaceStatus.VERIFIED,(0,len(x))),))
 def test_strict_geometry_and_immutable(self):
  a=self.s([[0,0,0],[1,0,0]]);r=qualify_surface_segments(a,coordinate_frame='w',units='m',timestamp=1,clock_domain='s',config=PathQualificationConfig(policy='strict',experimental_max_surface_uncertainty_m=1),uncertainty=np.array([.1,.1]));q=r.segments[0];self.assertEqual(q.status,PathQualificationStatus.VERIFIED);self.assertAlmostEqual(q.s_m[-1],1);self.assertFalse(q.xyz.flags.writeable);a.segments[0].xyz[0,0]=9;self.assertEqual(q.xyz[0,0],0)
 def test_gates(self):
  for x in ([[0,0,0],[0,0,0]],[[0,0,0],[1,0,1]]):self.assertNotEqual(qualify_surface_segments(self.s(x),coordinate_frame='w',units='m',timestamp=1,clock_domain='s').segments[0].status,PathQualificationStatus.VERIFIED)
