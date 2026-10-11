"""Experimental stateful Vision V1.1 runtime; V1 ``process(rgb)`` is untouched."""
from dataclasses import dataclass
import cv2
import numpy as np
from config import VisionConfig
from data_source import RGBDFrame
from .contracts import (ContourType, LineVisionResult, ObservationStatus, OverallState,
                        OrientationSource, PointDerivationMethod, TrackingState,
                        VisionPointProvenance, VisionResult)
from .tracking import LineTracker
from .v11 import resample_arc_length, resample_arc_length_with_lineage, sample_depth_valid, semantic_order
from .vision_processor import VisionProcessor

@dataclass
class _Memory:
    points: np.ndarray|None=None
    timestamp: float|None=None

class ExperimentalVisionRuntime:
    """ROI, tracking, expiry and reacquisition adapter for simulation research."""
    def __init__(self, config: VisionConfig):
        self.config=config; self.processor=VisionProcessor(config)
        self.left_tracker=LineTracker(config.experimental_tracking_timeout_ms,config.experimental_observation_hysteresis_ms)
        self.right_tracker=LineTracker(config.experimental_tracking_timeout_ms,config.experimental_observation_hysteresis_ms)
        self.left_memory=_Memory(); self.right_memory=_Memory(); self.roi=None; self._last_timestamp=None
    def _roi(self, shape, predicting=False):
        h,w=shape[:2]
        if self.roi is None: return (0,0,w,h)
        x0,y0,x1,y1=self.roi; pad=self.config.experimental_roi_margin_px+(self.config.experimental_roi_prediction_expansion_px if predicting else 0)
        return (max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad))
    def _candidates(self,rgb,roi):
        x0,y0,x1,y1=roi; crop=rgb[y0:y1,x0:x1]
        if crop.size==0:return []
        # Contrast is normalized only in experimental runtime; V1 pixels stay untouched.
        lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        normalized=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2RGB)
        hsv=cv2.cvtColor(normalized,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.array(self.config.hsv_lower),np.array(self.config.hsv_upper))
        k=np.ones((self.config.morphology_size,self.config.morphology_size),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        lines=[]
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA] < self.config.min_component_area: continue
            try: line=self.processor._trace(self.processor._thin(labels==label))
            except ValueError: continue
            lines.append(line+np.array([x0,y0]))
        return sorted(lines,key=lambda line:len(line),reverse=True)[:4]
    def _line(self, points, observed, tracker, memory, frame, line_id):
        timestamp=frame.timestamp
        status,state,age=tracker.update(observed,timestamp)
        if observed:
            raw_points=np.asarray(points,dtype=float)
            resampled=resample_arc_length_with_lineage(raw_points,self.config.experimental_depth_resample_points,
                closed=np.linalg.norm(raw_points[0]-raw_points[-1])<2)
            points=resampled.points
            valid,ratio,ranges=sample_depth_valid(frame.depth,points)
            quality=min(1.,cv2.Laplacian(cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()/max(self.config.experimental_blur_laplacian_min,1e-9))
            confidence=ratio*quality; memory.points=points; memory.timestamp=timestamp
            contour=ContourType.CLOSED if np.linalg.norm(points[0]-points[-1]) < 2 else ContourType.OPEN
            # Resampling changes the point identity.  The nearest raw trace index is
            # disclosed, but no external source-point identifier is invented.
            provenance=tuple(VisionPointProvenance(
                line_id=line_id,source_index=reference.start_index,pixel_xy=tuple(point),
                observation_status=status,tracking_state=state,
                derivation_method=PointDerivationMethod.RESAMPLING,
                lineage_available=reference.available,confidence_reference="line_confidence",
                timestamp=timestamp,clock_domain=frame.clock_domain,
                source_segment_start_index=reference.start_index,source_segment_end_index=reference.end_index,
                interpolation_fraction=reference.fraction,
                source_segment_start_xy=None if reference.start_index is None else tuple(raw_points[reference.start_index]),
                source_segment_end_xy=None if reference.end_index is None else tuple(raw_points[reference.end_index]))
                for point,reference in zip(points,resampled.lineage))
            diagnostics=() if resampled.corner_status=="PRESERVED" else ("RESAMPLING_CORNER_PRESERVATION_LIMITED",)
            return LineVisionResult(points,status,state,confidence,0.,valid,ratio,ranges,OrientationSource.REFERENCE,0.,confidence>=self.config.experimental_min_observation_confidence,(),contour,provenance,diagnostics)
        uncertainty=float('inf') if memory.timestamp is None else age/self.config.experimental_tracking_timeout_ms*self.config.experimental_roi_prediction_expansion_px
        if state is TrackingState.PREDICTING and memory.points is not None and uncertainty<=self.config.experimental_roi_prediction_expansion_px:
            provenance=tuple(VisionPointProvenance(line_id=line_id,source_index=None,pixel_xy=tuple(point),
                observation_status=status,tracking_state=state,derivation_method=PointDerivationMethod.PREDICTION,
                lineage_available=False,confidence_reference="line_confidence",timestamp=timestamp,
                clock_domain=frame.clock_domain) for point in memory.points)
            return LineVisionResult(memory.points,status,state,max(0.,1-age/self.config.experimental_tracking_timeout_ms)*.5,age,None,0.,(),OrientationSource.REFERENCE,uncertainty,False,((0,len(memory.points)),),ContourType.UNKNOWN,provenance)
        return LineVisionResult(None,ObservationStatus.INVALID,TrackingState.LOST,0.,age,None,0.,(),OrientationSource.AMBIGUOUS,float('inf'),False,(),ContourType.UNKNOWN)
    def process_frame(self, frame: RGBDFrame)->VisionResult:
        if frame.timestamp is None or not np.isfinite(frame.timestamp): raise ValueError('Experimental runtime requires a finite frame timestamp')
        if self._last_timestamp is not None and frame.timestamp <= self._last_timestamp: raise ValueError('Frame timestamp must increase')
        self._last_timestamp=frame.timestamp
        if frame.rgb.shape[:2]!=frame.depth.shape or not frame.rgb_depth_registered: raise ValueError('RGB-D frame is not registered')
        predicting=self.left_tracker.state is TrackingState.PREDICTING or self.right_tracker.state is TrackingState.PREDICTING
        roi=self._roi(frame.rgb.shape,predicting); candidates=self._candidates(frame.rgb,roi)
        left=right=None; source=OrientationSource.AMBIGUOUS
        if len(candidates)>=2:
            reference=self.left_memory.points
            left,right,source=semantic_order(candidates[0],candidates[1],reference_left=reference)
            if source is not OrientationSource.AMBIGUOUS:
                separation=np.linalg.norm(left.mean(0)-right.mean(0))
                if not self.config.experimental_min_line_separation_px<=separation<=self.config.experimental_max_line_separation_px: left=right=None
        elif len(candidates)==1:
            if self.left_memory.points is not None and self.right_memory.points is not None:
                dl=np.linalg.norm(candidates[0].mean(0)-self.left_memory.points.mean(0)); dr=np.linalg.norm(candidates[0].mean(0)-self.right_memory.points.mean(0))
                if abs(dl-dr)>1e-6:
                    left=candidates[0] if dl<dr else None; right=candidates[0] if dr<dl else None
        l=self._line(left,left is not None,self.left_tracker,self.left_memory,frame,"left"); r=self._line(right,right is not None,self.right_tracker,self.right_memory,frame,"right")
        observed=[l.observation_status is ObservationStatus.OBSERVED,r.observation_status is ObservationStatus.OBSERVED]
        overall=OverallState.BOTH_OK if all(observed) else OverallState.ONE_DEGRADED if any(observed) else OverallState.BOTH_LOST
        if all(observed):
            allpoints=np.vstack((l.points,r.points)); x0,y0=allpoints.min(0).astype(int); x1,y1=allpoints.max(0).astype(int); self.roi=(x0,y0,x1,y1)
        else: self.roi=None # missing observation always enables full-frame reacquisition
        return VisionResult(frame.timestamp,frame.clock_domain,None,frame_id=frame.frame_id,coordinate_frame=frame.coordinate_frame,
            camera_to_reference=frame.calibration.camera_to_world,occlusion_status='KNOWN_MASK' if frame.occlusion_mask is not None else 'UNKNOWN',fabric_movement_uncertainty=frame.fabric_movement_uncertainty,left=l,right=r,overall_state=overall)
