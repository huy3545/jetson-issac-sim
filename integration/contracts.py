"""Immutable records shared by the experimental Stage 7B integration layer.

The records deliberately describe evidence that is present; they never supply a
default coordinate frame, timestamp, provenance reference, or trust decision.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdmissionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class IntegrationRecord:
    run_id: str
    correlation_id: str
    stage_id: str
    admission_status: AdmissionStatus
    source_frame_id: str | int | None = None
    segment_id: str | None = None
    timestamp: float | None = None
    clock_domain: str | None = None
    coordinate_frame: str | None = None
    units: str | None = None
    config_version: str | None = None
    provenance_references: tuple[str, ...] = ()
    confidence: float | None = None
    geometry_status: str | None = None
    uncertainty_status: str | None = None
    qualification_status: str | None = None
    reason_codes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    execution_authorized: bool = False


@dataclass(frozen=True)
class IntegrationEvent:
    run_id: str
    correlation_id: str
    stage_id: str
    event_type: str
    timestamp: float | None
    reason_codes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationRunResult:
    run_id: str
    correlation_id: str
    records: tuple[IntegrationRecord, ...]
    events: tuple[IntegrationEvent, ...]
    final_status: AdmissionStatus
    audit_complete: bool
    execution_authorized: bool = False
    details: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
