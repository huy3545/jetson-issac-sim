import unittest,numpy as np
from path_planning.surface_mapping import SurfaceSegment,SurfaceStatus,SurfaceMappingResult,SourceProvenance
from path_planning.path_qualification import *
class Q(unittest.TestCase):
 def s(self,x):
  p=SourceProvenance('xy','w','m',1,'s','KNOWN'); h=SourceProvenance('heightmap','w','m',1,'s','KNOWN')
  return SurfaceMappingResult((SurfaceSegment(2,np.array(x,float),SurfaceStatus.VERIFIED,(0,len(x)),segment_id='2:0',source_indices=(0,len(x)),coordinate_frame='w',units='m',timestamp=1,clock_domain='s',surface_uncertainty=.1,uncertainty_status='KNOWN',provenance=h),),xy_provenance=p,surface_provenance=h,output_coordinate_frame='w',output_units='m')
 def test_strict_geometry_and_immutable(self):
  a=self.s([[0,0,0],[1,0,0]]);r=qualify_surface_segments(a,coordinate_frame='w',units='m',timestamp=1,clock_domain='s',config=PathQualificationConfig(policy='strict',experimental_max_surface_uncertainty_m=1),uncertainty=np.array([.1,.1]));q=r.segments[0];self.assertEqual(q.status,PathQualificationStatus.VERIFIED);self.assertAlmostEqual(q.s_m[-1],1);self.assertFalse(q.xyz.flags.writeable);a.segments[0].xyz[0,0]=9;self.assertEqual(q.xyz[0,0],0)
 def test_gates(self):
  for x in ([[0,0,0],[0,0,0]],[[0,0,0],[1,0,1]]):self.assertNotEqual(qualify_surface_segments(self.s(x),coordinate_frame='w',units='m',timestamp=1,clock_domain='s').segments[0].status,PathQualificationStatus.VERIFIED)
 def test_geometry_only_numeric_metadata_and_reverse(self):
  a=self.s([[0,0,0],[1,0,1],[2,0,2]]);c=PathQualificationConfig(policy='geometry_only',experimental_max_3d_step_m=2,experimental_max_z_jump_m=2)
  q=qualify_surface_segments(a,coordinate_frame='w',units='m',timestamp=1,clock_domain='s',config=c).segments[0];self.assertEqual(q.status,PathQualificationStatus.VERIFIED);self.assertEqual(q.uncertainty_status,UncertaintyStatus.KNOWN);self.assertAlmostEqual(q.s_m[-1],2*np.sqrt(2));self.assertTrue(np.isfinite(q.tangent_xyz).all());self.assertFalse(hasattr(q,'speed'))
 def test_source_invalid_provenance_step_uncertainty_and_runs(self):
  bad=SurfaceMappingResult((SurfaceSegment(4,np.array([[0,0,0],[1,0,0]]),SurfaceStatus.INVALID,(4,6)),));q=qualify_surface_segments(bad,coordinate_frame='w',units='m',timestamp=1,clock_domain='s',config=PathQualificationConfig(policy='geometry_only')).segments[0];self.assertEqual(q.status,PathQualificationStatus.INVALID);self.assertEqual(q.source_id,4);self.assertEqual(q.source_indices,(4,6))
  q=qualify_surface_segments(self.s([[0,0,0],[2,0,1]]),coordinate_frame='w',units='m',timestamp=1,clock_domain='s',config=PathQualificationConfig(policy='geometry_only',experimental_max_3d_step_m=1,experimental_max_z_jump_m=.5)).segments[0];self.assertIn('STEP_EXCEEDED',q.reason_codes);self.assertIn('Z_JUMP_EXCEEDED',q.reason_codes)
 def test_provenance_cannot_be_spoofed(self):
  q=qualify_surface_segments(self.s([[0,0,0],[1,0,0]]),coordinate_frame='other',units='m',timestamp=1,clock_domain='s',config=PathQualificationConfig(policy='geometry_only')).segments[0];self.assertIn('PROVENANCE_MISMATCH',q.reason_codes)
