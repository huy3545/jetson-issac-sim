"""Experimental Stage 7C-3 boundary proof; it never opens strict Path admission."""
from dataclasses import dataclass
from integration.contracts import AdmissionStatus
from integration.metric_bridge import MetricGeometryBridge, MetricBridgeVerifier
from path_planning.experimental_metric_projection import ExperimentalVisionMetricProjectionAdapter
from path_planning.vision_adapter import VisionPathAdapter

@dataclass(frozen=True)
class VisionPathQualificationResult:
    bridge_status:AdmissionStatus
    path_status:str
    reason_codes:tuple[str,...]
    execution_authorized:bool=False

class VisionPathQualificationHarness:
    """Calls public Vision → projection → bridge → Path APIs without trust upgrades."""
    def __init__(self):
        self.projector=ExperimentalVisionMetricProjectionAdapter()
        self.bridge=MetricGeometryBridge(); self.verifier=MetricBridgeVerifier()
        self.adapter=VisionPathAdapter(require_point_provenance=True)
    def evaluate(self,vision,depth,calibration,evidence,*,run_id,correlation_id):
        metric=self.projector.project(vision,depth,calibration,evidence)
        bridge=self.bridge.convert(vision,metric,run_id=run_id,correlation_id=correlation_id)
        reasons=list(self.verifier.verify_lineage(vision,metric,bridge))
        reasons.extend(bridge.reason_codes)
        # The Path adapter is deliberately called with the public Vision contract;
        # its pixel/camera rejection documents why no strict handoff can occur.
        path=self.adapter.plan(vision)
        reasons.extend(path.reason_codes)
        return VisionPathQualificationResult(bridge.admission_status,path.status.value,tuple(dict.fromkeys(reasons)))
