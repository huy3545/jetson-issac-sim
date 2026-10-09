"""Deterministic data-clock watchdog; runtime scheduling belongs in an adapter."""
from dataclasses import dataclass
from enum import Enum
from clock import Clock

class WatchdogState(str, Enum): HEALTHY="HEALTHY"; CAMERA_STALE="CAMERA_STALE"; MACHINE_STATE_STALE="MACHINE_STATE_STALE"; VISION_STALL="VISION_STALL"
@dataclass(frozen=True)
class WatchdogResult: state: WatchdogState; reason: str=""
class WatchdogCore:
    def __init__(self, experimental_stale_ms: float):
        self.stale=experimental_stale_ms/1000
        self._last_frame_timestamp = None
    def observe_frame_timestamp(self, timestamp: float) -> WatchdogResult:
        if self._last_frame_timestamp is not None and timestamp <= self._last_frame_timestamp:
            return WatchdogResult(WatchdogState.CAMERA_STALE, "NON_INCREASING_FRAME_TIMESTAMP")
        self._last_frame_timestamp=timestamp
        return WatchdogResult(WatchdogState.HEALTHY)
    def evaluate(self, now, last_frame, last_machine_state, last_vision, clock_advanced=True):
        if not clock_advanced: return WatchdogResult(WatchdogState.HEALTHY, "DATA_CLOCK_PAUSED")
        for state, seen, label in ((WatchdogState.CAMERA_STALE,last_frame,"FRAME"),(WatchdogState.MACHINE_STATE_STALE,last_machine_state,"MACHINE_STATE"),(WatchdogState.VISION_STALL,last_vision,"VISION")):
            if seen is None or now-seen>self.stale: return WatchdogResult(state, label+"_STALE")
        return WatchdogResult(WatchdogState.HEALTHY)

class ProcessHealthAdapter:
    """Optional process-health layer, deliberately separate from data-clock core."""
    def __init__(self, clock: Clock, experimental_stale_ms: float):
        self.clock=clock; self.threshold=experimental_stale_ms/1000; self.last_heartbeat=None
    def heartbeat(self) -> None: self.last_heartbeat=self.clock.now()
    def evaluate(self) -> WatchdogResult:
        if self.last_heartbeat is None or self.clock.now()-self.last_heartbeat > self.threshold:
            return WatchdogResult(WatchdogState.VISION_STALL,"PROCESS_HEARTBEAT_STALE")
        return WatchdogResult(WatchdogState.HEALTHY)
