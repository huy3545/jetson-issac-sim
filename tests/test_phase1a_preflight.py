"""Independent fixture preflight; this module deliberately does not import solver code."""
import json
import math
import hashlib
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "phase1a_open_channels.json"
ROOT = Path(__file__).parent.parent

class Phase1aFixturePreflight(unittest.TestCase):
    def test_locked_fixture_hashes(self):
        expected=json.loads((ROOT/'phase1a_fixture_hashes.json').read_text())
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),digest)
    def test_fixture_is_geometrically_consistent(self):
        data=json.loads(FIXTURE.read_text())
        self.assertEqual(data['units'], 'metres')
        parallel=next(c for c in data['cases'] if c['id']=='parallel_open')
        self.assertEqual(parallel['expected_status'],'CONVERGED')
        self.assertEqual(parallel['reference_centerline'][0][1],0.0)
        arc=next(c for c in data['cases'] if c['id']=='concentric_arc_open')
        self.assertLess(arc['r1'],arc['reference_radius']); self.assertLess(arc['reference_radius'],arc['r2'])
        self.assertGreater(arc['angle_end_rad'],arc['angle_start_rad'])
        self.assertGreaterEqual(arc['samples'],3)
        # Formula is independent of any solver-generated point.
        self.assertGreater((arc['angle_end_rad']-arc['angle_start_rad'])*arc['reference_radius'], .95*.1)
        self.assertEqual(set(data['mutation_detection_plan']), {'skip_postprojection','skip_inside_channel','skip_continuity','first_multiple_root','vertex_distance','loose_residual'})
    def test_crossing_case_has_intersection(self):
        data=json.loads(FIXTURE.read_text()); case=next(c for c in data['cases'] if c['id']=='crossing_boundaries')
        (a,b),(c,d)=case['left'],case['right']
        cross=lambda p,q,r: (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        self.assertLess(cross(a,b,c)*cross(a,b,d),0)
        self.assertLess(cross(c,d,a)*cross(c,d,b),0)
