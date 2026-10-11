"""Immutable, JSON-safe contracts for the Stage 7D-2 read-only preview."""
from dataclasses import dataclass
import json
import math

PREVIEW_SCHEMA_VERSION = "1.0"
MAX_PREVIEW_POINTS = 4096


def _points(value, name):
    if not isinstance(value, tuple) or not value or len(value) > MAX_PREVIEW_POINTS:
        raise ValueError(f"invalid {name} point count")
    clean=[]
    for point in value:
        if not isinstance(point, tuple) or len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError(f"invalid {name} coordinate")
        clean.append((float(point[0]), float(point[1])))
    return tuple(clean)


@dataclass(frozen=True)
class IsaacPreviewArtifact:
    schema_version: str; preview_id: str; run_id: str; correlation_id: str
    source_frame_id: str | int | None; timestamp: float; clock_domain: str
    coordinate_frame: str; units: str; synthetic_fixture_id: str
    left_boundary_xy: tuple[tuple[float, float], ...]
    right_boundary_xy: tuple[tuple[float, float], ...]
    centerline_xy: tuple[tuple[float, float], ...] | None
    geometry_status: str; solver_diagnostics: tuple[str, ...]
    source_provenance_status: str; display_transform: dict
    preview_authorized: bool; execution_authorized: bool; reason_codes: tuple[str, ...]

    def __post_init__(self):
        if self.schema_version != PREVIEW_SCHEMA_VERSION or not self.preview_id or not self.run_id or not self.correlation_id:
            raise ValueError("invalid preview identity")
        if self.units != "m" or not self.coordinate_frame or self.coordinate_frame in ("camera", "legacy"):
            raise ValueError("preview requires metric reference geometry")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp) or not self.clock_domain:
            raise ValueError("invalid preview time")
        object.__setattr__(self, "left_boundary_xy", _points(self.left_boundary_xy, "left"))
        object.__setattr__(self, "right_boundary_xy", _points(self.right_boundary_xy, "right"))
        if self.centerline_xy is not None: object.__setattr__(self, "centerline_xy", _points(self.centerline_xy, "centerline"))
        if self.execution_authorized: raise ValueError("preview cannot authorize execution")
        if self.preview_authorized != (self.centerline_xy is not None): raise ValueError("preview authorization mismatch")
        if self.preview_authorized and self.geometry_status != "SYNTHETIC_GEOMETRY_VERIFIED":
            raise ValueError("verified centerline requires qualified geometry")
        if not isinstance(self.display_transform, dict) or self.display_transform.get("kind") != "DISPLAY_ONLY_XY_TO_XYZ":
            raise ValueError("missing display-only transform")

    def to_dict(self):
        return {"schema_version": self.schema_version, "preview_id": self.preview_id, "run_id": self.run_id,
            "correlation_id": self.correlation_id, "source_frame_id": self.source_frame_id,
            "timestamp": self.timestamp, "clock_domain": self.clock_domain,
            "coordinate_frame": self.coordinate_frame, "units": self.units,
            "synthetic_fixture_id": self.synthetic_fixture_id,
            "left_boundary_xy": self.left_boundary_xy, "right_boundary_xy": self.right_boundary_xy,
            "centerline_xy": self.centerline_xy, "geometry_status": self.geometry_status,
            "solver_diagnostics": self.solver_diagnostics, "source_provenance_status": self.source_provenance_status,
            "display_transform": self.display_transform, "preview_authorized": self.preview_authorized,
            "execution_authorized": False, "reason_codes": self.reason_codes}

    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, payload):
        raw=json.loads(payload)
        if not isinstance(raw, dict): raise ValueError("preview JSON must be an object")
        for key in ("left_boundary_xy", "right_boundary_xy", "centerline_xy"):
            if raw.get(key) is not None: raw[key] = tuple(tuple(point) for point in raw[key])
        for key in ("solver_diagnostics", "reason_codes"):
            raw[key] = tuple(raw.get(key, ()))
        return cls(**raw)
