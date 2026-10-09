"""Bounded, deterministic timestamp matching for machine state samples."""
from dataclasses import dataclass
from enum import Enum
from collections import deque

class SyncMode(str, Enum):
    INTERPOLATED = "INTERPOLATED"
    EXTRAPOLATED = "EXTRAPOLATED"
    INVALID = "INVALID"

@dataclass(frozen=True)
class MachineState:
    x: float; y: float; z: float; c: float
    timestamp: float; clock_domain: str

@dataclass(frozen=True)
class SyncResult:
    machine_state_at_frame: MachineState | None
    sync_mode: SyncMode
    sync_age_ms: float | None
    time_sync_valid: bool
    reason: str = ""

class StateSyncBuffer:
    def __init__(self, capacity: int = 64): self._states = deque(maxlen=capacity)
    def add(self, state: MachineState) -> None:
        if self._states and state.clock_domain != self._states[-1].clock_domain:
            self._states.clear()
        if self._states and state.timestamp <= self._states[-1].timestamp:
            raise ValueError("MachineState timestamp must increase")
        self._states.append(state)
    def match(self, timestamp: float, clock_domain: str, experimental_extrapolation_ms: float) -> SyncResult:
        if not self._states: return SyncResult(None, SyncMode.INVALID, None, False, "INSUFFICIENT_STATE")
        if clock_domain != self._states[-1].clock_domain:
            return SyncResult(None, SyncMode.INVALID, None, False, "CLOCK_DOMAIN_MISMATCH")
        states = list(self._states)
        for a, b in zip(states, states[1:]):
            if a.timestamp <= timestamp <= b.timestamp:
                ratio = (timestamp-a.timestamp)/(b.timestamp-a.timestamp)
                vals = [getattr(a,k)+(getattr(b,k)-getattr(a,k))*ratio for k in "xyzc"]
                return SyncResult(MachineState(*vals, timestamp, clock_domain), SyncMode.INTERPOLATED, 0.0, True)
        anchor = states[0] if timestamp < states[0].timestamp else states[-1]
        age_ms = abs(timestamp-anchor.timestamp)*1000
        if age_ms > experimental_extrapolation_ms:
            return SyncResult(None, SyncMode.INVALID, age_ms, False, "EXTRAPOLATION_LIMIT")
        return SyncResult(MachineState(anchor.x, anchor.y, anchor.z, anchor.c, timestamp, clock_domain),
                          SyncMode.EXTRAPOLATED, age_ms, True, "HOLD_LAST_STATE")
