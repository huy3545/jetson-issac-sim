"""Clock domains are explicit so simulation time never masquerades as wall time."""
from dataclasses import dataclass
import time
from typing import Protocol

class Clock(Protocol):
    @property
    def domain_id(self) -> str: ...
    def now(self) -> float: ...

@dataclass
class FakeClock:
    value: float = 0.0
    domain_id: str = "simulation"
    def now(self) -> float: return self.value
    def advance(self, seconds: float) -> None: self.value += seconds

@dataclass(frozen=True)
class MonotonicClock:
    domain_id: str = "process_monotonic"
    def now(self) -> float: return time.monotonic()
