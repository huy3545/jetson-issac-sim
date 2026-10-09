import unittest
import numpy as np

from path_planning.legacy_geometry_gate import (
    GeometryEvidence, GeometryGateStatus, LegacyGeometryGate, LegacyGeometryGateConfig,
)


def corridor(center, width=.05):
    tangent = np.gradient(center, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return center-normal*width, center+normal*width


def evidence(reference):
    return GeometryEvidence("workpiece_m", "m", 1., "simulation", "synthetic", True, True, reference)


class LegacyGeometryGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = LegacyGeometryGate(LegacyGeometryGateConfig(
            max_reference_deviation_m=1e-5, max_reference_rms_m=1e-5))
        x = np.linspace(0., .3, 31)
        self.center = np.c_[x, np.zeros_like(x)]
        self.left, self.right = corridor(self.center)

    def test_accepts_evidenced_open_channel_and_is_order_invariant(self):
        for left, right in ((self.left, self.right), (self.left, self.right[::-1]), (self.right, self.left)):
            result = self.gate.evaluate(left, right, evidence(self.center))
            self.assertEqual(result.status, GeometryGateStatus.ACCEPT)
            self.assertLessEqual(result.max_reference_deviation_m, 1e-12)

    def test_missing_evidence_and_ambiguous_endpoint_pairing_are_uncertain(self):
        result = self.gate.evaluate(self.left, self.right)
        self.assertEqual(result.status, GeometryGateStatus.UNCERTAIN)
        self.assertIn("MISSING_GEOMETRY_METADATA", result.reason_codes)
        ambiguous = LegacyGeometryGate(LegacyGeometryGateConfig(
            correspondence_ambiguity_m=10., max_reference_deviation_m=1e-5, max_reference_rms_m=1e-5))
        result = ambiguous.evaluate(self.left, self.right, evidence(self.center))
        self.assertEqual(result.status, GeometryGateStatus.UNCERTAIN)
        self.assertIn("AMBIGUOUS_ENDPOINT_CORRESPONDENCE", result.reason_codes)

    def test_rejects_invalid_boundaries_and_topology(self):
        duplicate = self.left.copy(); duplicate[3] = duplicate[2]
        self.assertEqual(self.gate.evaluate(duplicate, self.right, evidence(self.center)).status, GeometryGateStatus.REJECT)
        bow = np.array([[0., 0.], [.2, .2], [0., .2], [.2, 0.]])
        self.assertIn("BOUNDARY_SELF_INTERSECTION", self.gate.evaluate(bow, self.right, evidence(self.center)).reason_codes)
        crossing = np.array([[0., -.1], [.3, .1]])
        self.assertIn("CHANNEL_SELF_INTERSECTION", self.gate.evaluate(self.left, crossing, evidence(self.center)).reason_codes)

    def test_reference_deviation_rejects_instead_of_accepting(self):
        wrong = self.center + [0., .03]
        result = self.gate.evaluate(self.left, self.right, evidence(wrong))
        self.assertEqual(result.status, GeometryGateStatus.REJECT)
        self.assertIn("REFERENCE_DEVIATION_EXCEEDED", result.reason_codes)

