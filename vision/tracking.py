"""Independent line FSM with timestamp-based hysteresis; no wall-clock timers."""
from dataclasses import dataclass
from .contracts import ObservationStatus, TrackingState

@dataclass
class LineTracker:
    timeout_ms: float
    hysteresis_ms: float
    state: TrackingState = TrackingState.LOST
    last_observed: float | None = None
    valid_since: float | None = None
    def update(self, observed: bool, timestamp: float) -> tuple[ObservationStatus, TrackingState, float]:
        if observed:
            if self.valid_since is None: self.valid_since=timestamp
            self.last_observed=timestamp
            if timestamp-self.valid_since >= self.hysteresis_ms/1000:
                self.state=TrackingState.TRACKING
            return ObservationStatus.OBSERVED,self.state,0.0
        self.valid_since=None
        if self.last_observed is None:
            self.state=TrackingState.LOST; return ObservationStatus.INVALID,self.state,float("inf")
        age=(timestamp-self.last_observed)*1000
        self.state=TrackingState.PREDICTING if age <= self.timeout_ms else TrackingState.LOST
        return ObservationStatus.OCCLUDED if self.state is TrackingState.PREDICTING else ObservationStatus.INVALID,self.state,age
