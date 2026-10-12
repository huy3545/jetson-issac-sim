import unittest
from integration.isaac_preview_contracts import IsaacPreviewArtifact, PREVIEW_SCHEMA_VERSION
from simulation.isaac_scene_validator import validate_preview_scene

class SceneValidatorOfflineTests(unittest.TestCase):
    def _artifact(self):
        return IsaacPreviewArtifact(PREVIEW_SCHEMA_VERSION,"preview","run","corr","frame",0.,"sim","workpiece_m","m","stage7c6-square-miter-90",((0.,.1),(.5,.1)),((0.,-.1),(.7,-.1)),((0.,0.),(.6,.6)),"SYNTHETIC_GEOMETRY_VERIFIED",(),"POINT_PROVENANCE_AVAILABLE",{"kind":"DISPLAY_ONLY_XY_TO_XYZ","usd_up_axis":"Z","z_preview_m":0.,"scale_m_per_stage_unit":1.,"physical_transform_verified":False},True,False,("STRICT_GATE_B_BLOCKED",))
    def test_invalid_tolerance_fails_before_usd_access(self):
        result=validate_preview_scene(None,self._artifact(),tolerance=True)
        self.assertEqual(result.status,"FAIL"); self.assertIn("INVALID_TOLERANCE",result.reason_codes)
    def test_malformed_expected_geometry_fails_before_usd_access(self):
        bad=object.__new__(IsaacPreviewArtifact)
        result=validate_preview_scene(None,bad)
        self.assertEqual(result.status,"FAIL"); self.assertIn("USD_STAGE_INVALID",result.reason_codes)
    def test_missing_stage_fails_closed_without_usd_runtime(self):
        result=validate_preview_scene(None,self._artifact())
        self.assertEqual(result.status,"FAIL"); self.assertIn("USD_STAGE_INVALID",result.reason_codes)

if __name__=="__main__": unittest.main()
