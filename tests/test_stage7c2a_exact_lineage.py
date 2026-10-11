"""Stage 7C-2A exact resampling and independent projection checks."""
from dataclasses import replace
import unittest
import numpy as np

from data_source import CameraCalibration
from integration.contracts import AdmissionStatus
from integration.metric_bridge import MetricBridgeVerifier, MetricGeometryBridge
from path_planning.experimental_metric_projection import ExperimentalVisionMetricProjectionAdapter, TransformEvidence, TransformValidity
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource, PointDerivationMethod,
                              TrackingState, VisionPointProvenance, VisionResult)
from vision.v11 import resample_arc_length, resample_arc_length_with_lineage


class ExactLineageTests(unittest.TestCase):
    def test_legacy_resampling_output_is_unchanged(self):
        raw=np.array([[0.,0.],[2.,0.],[2.,3.]])
        sampled=resample_arc_length_with_lineage(raw,6)
        np.testing.assert_allclose(sampled.points,resample_arc_length(raw,6))
        for pixel,reference in zip(sampled.points,sampled.lineage):
            self.assertTrue(reference.available)
            np.testing.assert_allclose(raw[reference.start_index]*(1-reference.fraction)+raw[reference.end_index]*reference.fraction,pixel)

    def test_corner_matrix_and_degenerate_disclosure(self):
        cases={
            "straight":np.array([[0.,0.],[4.,0.]]),
            "right_90":np.array([[0.,0.],[2.,0.],[2.,2.]]),
            "reflex_270":np.array([[0.,0.],[2.,0.],[2.,-2.],[0.,-2.]]),
            "turnback_180":np.array([[0.,0.],[2.,0.],[0.,0.]]),
            "large_curve":np.c_[10*np.cos(np.linspace(0,np.pi/2,9)),10*np.sin(np.linspace(0,np.pi/2,9))],
            "small_curve":np.c_[.2*np.cos(np.linspace(0,np.pi/2,9)),.2*np.sin(np.linspace(0,np.pi/2,9))],
            "uneven":np.array([[0.,0.],[.01,0.],[4.,0.],[4.,3.]]),
            "overlap":np.array([[0.,0.],[0.,0.],[3.,0.]])}
        for name,raw in cases.items():
            result=resample_arc_length_with_lineage(raw,3)
            self.assertEqual(len(result.points),3,name)
            self.assertTrue(all(item.available for item in result.lineage),name)
        limited=resample_arc_length_with_lineage(cases["right_90"],2)
        self.assertEqual(limited.corner_status,"LIMITED")
        self.assertEqual(limited.omitted_corner_indices,(1,))
        degenerate=resample_arc_length_with_lineage(np.array([[1.,1.],[1.,1.]]),3)
        self.assertEqual(degenerate.corner_status,"UNAVAILABLE")
        self.assertTrue(all(not item.available for item in degenerate.lineage))

    def _fixture(self):
        raw_left=np.array([[1.,1.],[3.,1.],[3.,3.]])
        raw_right=raw_left+np.array([3.,0.])
        matrix=np.eye(4); depth=np.arange(64,dtype=float).reshape(8,8)/100+.5
        def line(name,raw):
            sampled=resample_arc_length_with_lineage(raw,4)
            provenance=tuple(VisionPointProvenance(name,ref.start_index,tuple(pixel),ObservationStatus.OBSERVED,
                TrackingState.TRACKING,PointDerivationMethod.RESAMPLING,lineage_available=ref.available,
                timestamp=1.,clock_domain="sim",source_segment_start_index=ref.start_index,
                source_segment_end_index=ref.end_index,interpolation_fraction=ref.fraction,
                source_segment_start_xy=None if ref.start_index is None else tuple(raw[ref.start_index]),
                source_segment_end_xy=None if ref.end_index is None else tuple(raw[ref.end_index]))
                for pixel,ref in zip(sampled.points,sampled.lineage))
            return LineVisionResult(sampled.points,ObservationStatus.OBSERVED,TrackingState.TRACKING,.8,
                depth_valid=np.ones(len(sampled.points),bool),orientation_source=OrientationSource.REFERENCE,
                verified=True,point_provenance=provenance)
        vision=VisionResult(1.,"sim",frame_id="f",coordinate_frame="camera",units="pixels",camera_to_reference=matrix,left=line("left",raw_left),right=line("right",raw_right))
        evidence=TransformEvidence("camera","workpiece_m","m","intr","v1","extr","v1","synthetic","synthetic",0.,0.,1.,"sim","f",matrix,TransformValidity.SYNTHETIC)
        metric=ExperimentalVisionMetricProjectionAdapter().project(vision,depth,CameraCalibration(10.,10.,0.,0.,matrix),evidence)
        bridge=MetricGeometryBridge().convert(vision,metric,run_id="run",correlation_id="corr")
        return vision,metric,bridge,depth,CameraCalibration(10.,10.,0.,0.,matrix),evidence

    def test_independent_projection_verifier_and_fault_injection(self):
        vision,metric,bridge,depth,camera,evidence=self._fixture(); verifier=MetricBridgeVerifier()
        self.assertEqual(verifier.verify_projection(vision,metric,bridge,depth,camera,evidence),())
        point=metric.segments[0].point_provenance[0]
        bad_weight=replace(point,depth_samples=((point.depth_samples[0][0],point.depth_samples[0][1],.9),)+point.depth_samples[1:])
        bad_segment=replace(metric.segments[0],point_provenance=(bad_weight,)+metric.segments[0].point_provenance[1:])
        bad_metric=replace(metric,segments=(bad_segment,metric.segments[1]))
        self.assertIn("DEPTH_WEIGHT_MISMATCH",verifier.verify_projection(vision,bad_metric,bridge,depth,camera,evidence))
        bad_coordinate=replace(point,depth_samples=((7,7,point.depth_samples[0][2]),)+point.depth_samples[1:])
        coordinate_segment=replace(metric.segments[0],point_provenance=(bad_coordinate,)+metric.segments[0].point_provenance[1:])
        self.assertIn("DEPTH_SAMPLE_COORDINATE_MISMATCH",verifier.verify_projection(vision,replace(metric,segments=(coordinate_segment,metric.segments[1])),bridge,depth,camera,evidence))
        bad_xyz=replace(metric.segments[0],xyz=np.zeros_like(metric.segments[0].xyz))
        self.assertIn("PROJECTED_XYZ_MISMATCH",verifier.verify_projection(vision,replace(metric,segments=(bad_xyz,metric.segments[1])),bridge,depth,camera,evidence))
        bad_lineage=replace(point,vision_point=replace(point.vision_point,interpolation_fraction=.123))
        changed=replace(metric.segments[0],point_provenance=(bad_lineage,)+metric.segments[0].point_provenance[1:])
        self.assertIn("RESAMPLING_INTERPOLATION_MISMATCH",verifier.verify_projection(vision,replace(metric,segments=(changed,metric.segments[1])),bridge,depth,camera,evidence))
        self.assertIn("TRANSFORM_EVIDENCE_MISMATCH",verifier.verify_projection(vision,metric,bridge,depth,camera,replace(evidence,extrinsics_id="wrong")))
        reversed_bridge=replace(bridge,segments=tuple(reversed(bridge.segments)))
        self.assertIn("SEGMENT_ORDER_MISMATCH",verifier.verify_lineage(vision,metric,reversed_bridge))

    def test_missing_lineage_and_trust_constraints(self):
        vision,metric,bridge,depth,camera,evidence=self._fixture()
        self.assertEqual(bridge.admission_status,AdmissionStatus.BLOCKED)
        self.assertFalse(bridge.execution_authorized)
        unavailable=replace(vision.left,point_provenance=None)
        missing=replace(vision,left=unavailable)
        self.assertEqual(ExperimentalVisionMetricProjectionAdapter().project(missing,depth,camera,evidence).status.value,"UNCERTAIN")


if __name__=="__main__": unittest.main()
