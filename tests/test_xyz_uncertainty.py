import unittest
import numpy as np

from data_source import CameraCalibration
from path_planning.calibration_evidence import CalibrationEvidence, CalibrationTrustedProjectionAdapter, CalibrationTrust
from path_planning.experimental_metric_projection import TransformEvidence, TransformValidity
from path_planning.xyz_uncertainty import (ExperimentalXYZUncertaintyPropagator, NumericalPropagationStatus,
    UncertaintyEvidence, UncertaintyValidity, extrinsic_jacobian, pinhole_jacobian)
from vision.contracts import LineVisionResult, ObservationStatus, OrientationSource, TrackingState, VisionResult


class XYZUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.matrix=np.eye(4); self.camera=CameraCalibration(100.,120.,50.,30.,self.matrix); self.depth=np.full((80,100),.5)
        line=lambda x:LineVisionResult(np.array([[x+.25,20.5],[x+.25,30.5]]),ObservationStatus.OBSERVED,TrackingState.TRACKING,1.,0.,np.ones(2,bool),orientation_source=OrientationSource.REFERENCE,verified=True)
        self.vision=VisionResult(10.,'sim',frame_id='f',coordinate_frame='camera',units='pixels',camera_to_reference=self.matrix,left=line(40.),right=line(60.))
        self.transform=TransformEvidence('camera','workpiece_m','m','i','v','e','v','fixture','fixture',9.,9.,10.,'sim','f',self.matrix,TransformValidity.SYNTHETIC)
        self.cal_evidence=CalibrationEvidence('cal','v','cam','camera','workpiece_m','i','v','e','v','fixture','synthetic',9.,'sim','m','f',declared_trust=CalibrationTrust.SYNTHETIC)
        self.projection=CalibrationTrustedProjectionAdapter().project(self.vision,self.depth,self.camera,self.transform,self.cal_evidence)

    def _evidence(self, **updates):
        values=dict(source_id='f',camera_frame='camera',reference_frame='workpiece_m',units='m',observation_timestamp=10.,clock_domain='sim',uncertainty_source='synthetic',uncertainty_method='jacobian',uncertainty_version='v1',calibration_evidence_reference='cal@v',pixel_covariance=np.diag([.04,.09]),depth_sample_covariance=np.eye(4)*.01,intrinsics_covariance=np.eye(4)*1e-4,extrinsics_covariance=np.eye(6)*1e-5,zero_cross_covariance_confirmed=True,uncertainty_validity=UncertaintyValidity.SYNTHETIC)
        values.update(updates); return UncertaintyEvidence(**values)

    def test_analytic_pinhole_jacobian_matches_finite_difference(self):
        q=np.array([43.2, 27.4, .7, 100.,120.,50.,30.]); analytic=pinhole_jacobian(*q)
        def f(x): return np.array([(x[0]-x[5])*x[2]/x[3],(x[1]-x[6])*x[2]/x[4],x[2]])
        numeric=np.column_stack([(f(q+np.eye(7)[i]*1e-6)-f(q-np.eye(7)[i]*1e-6))/(2e-6) for i in range(7)])
        np.testing.assert_allclose(analytic,numeric,rtol=2e-6,atol=2e-6)

    def test_identity_rotation_translation_and_psd_covariance(self):
        for matrix in (np.eye(4), np.array([[0.,-1.,0.,.1],[1.,0.,0.,-.2],[0.,0.,1.,.3],[0.,0.,0.,1.]])):
            transform=TransformEvidence('camera','workpiece_m','m','i','v','e','v','fixture','fixture',9.,9.,10.,'sim','f',matrix,TransformValidity.SYNTHETIC)
            vision=VisionResult(10.,'sim',frame_id='f',coordinate_frame='camera',units='pixels',camera_to_reference=matrix,left=self.vision.left,right=self.vision.right)
            projection=CalibrationTrustedProjectionAdapter().project(vision,self.depth,self.camera,transform,self.cal_evidence)
            result=ExperimentalXYZUncertaintyPropagator().propagate(projection,vision,self.depth,self.camera,self._evidence())
            self.assertEqual(result.propagation_status,NumericalPropagationStatus.MODELLED_SYNTHETIC)
            self.assertTrue(all(np.linalg.eigvalsh(s.covariance_xyz).min()>=-1e-10 for s in result.segments))
            self.assertFalse(result.execution_authorized)

    def test_bilinear_correlated_depth_and_extrinsic_jacobian(self):
        independent=self._evidence(depth_sample_covariance=np.eye(4)*.01)
        correlated=self._evidence(depth_sample_covariance=np.full((4,4),.01))
        p=ExperimentalXYZUncertaintyPropagator(); a=p.propagate(self.projection,self.vision,self.depth,self.camera,independent); b=p.propagate(self.projection,self.vision,self.depth,self.camera,correlated)
        self.assertGreater(b.segments[0].covariance_xyz[0,2,2],a.segments[0].covariance_xyz[0,2,2])
        np.testing.assert_allclose(extrinsic_jacobian([1.,2.,3.])[:,3:],-np.array([[0.,-3.,2.],[3.,0.,-1.],[-2.,1.,0.]]))

    def test_invalid_covariance_and_metadata_reject_without_zeroing(self):
        cases=[self._evidence(pixel_covariance=np.eye(3)),self._evidence(pixel_covariance=np.array([[1.,1.],[0.,1.]])),self._evidence(pixel_covariance=np.diag([1.,-1.])),self._evidence(pixel_covariance=np.full((2,2),np.nan)),self._evidence(source_id='other'),self._evidence(zero_cross_covariance_confirmed=False)]
        for evidence in cases:
            result=ExperimentalXYZUncertaintyPropagator().propagate(self.projection,self.vision,self.depth,self.camera,evidence)
            self.assertEqual(result.propagation_status,NumericalPropagationStatus.INVALID); self.assertFalse(result.segments)

    def test_missing_or_declared_evidence_remains_unknown(self):
        p=ExperimentalXYZUncertaintyPropagator()
        missing=p.propagate(self.projection,self.vision,self.depth,self.camera,None)
        declared=p.propagate(self.projection,self.vision,self.depth,self.camera,self._evidence(uncertainty_validity=UncertaintyValidity.DECLARED))
        self.assertEqual(missing.propagation_status,NumericalPropagationStatus.INVALID)
        self.assertEqual(declared.propagation_status,NumericalPropagationStatus.UNKNOWN)
        self.assertFalse(declared.execution_authorized)

    def test_monte_carlo_matches_linearized_covariance_and_is_immutable(self):
        result=ExperimentalXYZUncertaintyPropagator().propagate(self.projection,self.vision,self.depth,self.camera,self._evidence(intrinsics_covariance=np.zeros((4,4)),extrinsics_covariance=np.zeros((6,6))))
        predicted=result.segments[0].covariance_xyz[0]
        rng=np.random.default_rng(17); n=30000; point=self.vision.left.points[0]
        # At (u,v)=(40.25,20.5), bilinear weights are [.375,.125,.375,.125],
        # so independent depth variance is .01 * sum(weights**2) = .003125.
        uv=rng.multivariate_normal(point,np.diag([.04,.09]),n); z=rng.normal(.5,np.sqrt(.003125),n)
        xyz=np.column_stack(((uv[:,0]-50)*z/100,(uv[:,1]-30)*z/120,z)); empirical=np.cov(xyz,rowvar=False)
        self.assertLess(np.max(np.abs(empirical-predicted)/np.maximum(np.abs(predicted),1e-12)),.12)
        self.assertFalse(result.segments[0].covariance_xyz.flags.writeable)
        with self.assertRaises(ValueError): result.segments[0].covariance_xyz[0,0,0]=0

    def test_provenance_is_preserved_and_no_downstream_dependencies_exist(self):
        result=ExperimentalXYZUncertaintyPropagator().propagate(self.projection,self.vision,self.depth,self.camera,self._evidence())
        self.assertEqual(result.source_id,'f'); self.assertEqual(result.reference_frame,'workpiece_m')
        self.assertEqual(result.segments[0].source_indices,(0,2))
        import path_planning.xyz_uncertainty as module
        with open(module.__file__,encoding='utf-8') as stream: source=stream.read()
        for forbidden in ('from motion_planning','from safety','from machine_controller','from data_logger','from .path_planner','legacy_geometry_gate'):
            self.assertNotIn(forbidden,source)


if __name__=='__main__': unittest.main()
