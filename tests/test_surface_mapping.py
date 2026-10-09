import unittest
import numpy as np
from path_planning.advanced_geometry import AdvancedGeometryPlanner
from path_planning.surface_mapping import MetricHeightmap,SurfaceMapper,SurfaceStatus
class SurfaceTests(unittest.TestCase):
 def setUp(self):
  x=np.linspace(0,1,31);self.g=AdvancedGeometryPlanner().plan_partial(np.c_[x,np.zeros_like(x)-.05],np.c_[x,np.zeros_like(x)+.05],np.ones(31,bool),np.ones(31,bool)).segments
 def h(self,z,**kw):return MetricHeightmap(np.asarray(z,float),(0,-1),(.1,.1),'workpiece_m','m',1.,**kw)
 def test_plane_bilinear_and_curved(self):
  y,x=np.mgrid[:30,:30];h=self.h(1+.1*x+.2*y);o=SurfaceMapper().map_segments(self.g,h,'workpiece_m',1.);self.assertTrue(all(s.status is SurfaceStatus.VERIFIED for s in o.segments));self.assertTrue(np.isfinite(o.segments[0].xyz).all())
 def test_invalid_missing_frame_time_uncertainty(self):
  z=np.ones((30,30));z[10:20,10:20]=np.nan
  for h,frame,time in ((self.h(z),'workpiece_m',1.),(self.h(np.ones((30,30))),'camera',1.),(self.h(np.ones((30,30))),'workpiece_m',2.)):
   o=SurfaceMapper().map_segments(self.g,h,frame,time);self.assertTrue(o.reason_codes or any(s.status is SurfaceStatus.INVALID or s.reason_codes for s in o.segments))
 def test_mask_partial_never_bridges(self):
  mask=np.ones((30,30),bool);mask[:,4:7]=False;o=SurfaceMapper().map_segments(self.g,self.h(np.ones((30,30)),valid_mask=mask),'workpiece_m',1.);self.assertGreaterEqual(len(o.segments),2);self.assertTrue(all(len(s.xyz)>1 for s in o.segments))
