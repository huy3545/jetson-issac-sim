"""Phase 3 immutable 3D path handoff; contains no motion commands."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .surface_mapping import SurfaceStatus

class PathQualificationStatus(str, Enum):
    VERIFIED = "VERIFIED"; INVALID = "INVALID"; UNCERTAIN = "UNCERTAIN"
class UncertaintyStatus(str, Enum):
    KNOWN = "KNOWN"; UNKNOWN = "UNKNOWN"; EXCEEDED = "EXCEEDED"
@dataclass(frozen=True)
class PathQualificationConfig:
    policy: str = "strict"
    experimental_max_3d_step_m: float | None = None
    experimental_max_z_jump_m: float | None = None
    experimental_max_surface_uncertainty_m: float | None = None
@dataclass(frozen=True)
class QualifiedPathSegment:
    segment_id: str; source_id: int; child_id: int; source_indices: tuple[int, int]
    xyz: np.ndarray; s_m: np.ndarray; tangent_xyz: np.ndarray; curvature_m_inv: np.ndarray
    uncertainty_status: UncertaintyStatus; status: PathQualificationStatus
    coordinate_frame: str; units: str; timestamp: float; clock_domain: str
    reason_codes: tuple[str, ...] = ()
@dataclass(frozen=True)
class QualifiedPathResult:
    schema_version: str; segments: tuple[QualifiedPathSegment, ...]; reason_codes: tuple[str, ...] = ()
def _ro(value):
    copied = np.array(value, dtype=float, copy=True); copied.setflags(write=False); return copied
def qualify_surface_segments(surface_result, *, coordinate_frame, units, timestamp, clock_domain, config=PathQualificationConfig(), uncertainty=None):
    """Qualify expected metadata against source metadata; no caller override is trusted.

    ``uncertainty`` remains only for V1 API compatibility and cannot upgrade source
    uncertainty into a verified result.
    """
    if config.policy not in ("strict", "geometry_only"): raise ValueError("unknown qualification policy")
    output = []
    for child, segment in enumerate(surface_result.segments):
        xyz = np.asarray(segment.xyz, dtype=float); codes = []
        frame = segment.coordinate_frame or surface_result.output_coordinate_frame
        source_units = segment.units or surface_result.output_units
        source_time, domain = segment.timestamp, segment.clock_domain
        if coordinate_frame != frame or units != source_units or timestamp != source_time or clock_domain != domain: codes.append("PROVENANCE_MISMATCH")
        if segment.status is not SurfaceStatus.VERIFIED: codes.append("UNVERIFIED_GEOMETRY")
        if source_units != "m" or not frame or not domain or source_time is None or not np.isfinite(source_time): codes.append("INVALID_PROVENANCE")
        if config.policy == "strict" and (surface_result.xy_provenance is None or segment.provenance is None): codes.append("PROVENANCE_MISSING")
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 2 or not np.isfinite(xyz).all(): codes.append("INVALID_XYZ")
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if xyz.ndim == 2 and len(xyz) > 1 else np.array([])
        if len(ds) and (ds <= 0).any(): codes.append("CONSECUTIVE_DUPLICATE")
        if config.experimental_max_3d_step_m is not None and len(ds) and (ds > config.experimental_max_3d_step_m).any(): codes.append("STEP_EXCEEDED")
        if config.experimental_max_z_jump_m is not None and len(xyz) > 1 and (np.abs(np.diff(xyz[:, 2])) > config.experimental_max_z_jump_m).any(): codes.append("Z_JUMP_EXCEEDED")
        # Z-only heightmap uncertainty is not proof of full XYZ accuracy.
        source_uncertainty = segment.surface_uncertainty
        uncertainty_known = segment.uncertainty_status == "KNOWN" and source_uncertainty is not None
        uncertainty_status = UncertaintyStatus.KNOWN if uncertainty_known else UncertaintyStatus.UNKNOWN
        if uncertainty_known and not np.isfinite(source_uncertainty): codes.append("INVALID_UNCERTAINTY")
        if config.policy == "strict" and (not uncertainty_known or config.experimental_max_surface_uncertainty_m is None): codes.append("UNCERTAINTY_UNCONFIGURED")
        if uncertainty_known and config.experimental_max_surface_uncertainty_m is not None and source_uncertainty > config.experimental_max_surface_uncertainty_m:
            codes.append("SURFACE_UNCERTAINTY_EXCEEDED"); uncertainty_status = UncertaintyStatus.EXCEEDED
        if codes:
            status = PathQualificationStatus.UNCERTAIN if any("UNCERTAINTY" in code or code == "PROVENANCE_MISSING" for code in codes) else PathQualificationStatus.INVALID
        else: status = PathQualificationStatus.VERIFIED
        s_m = np.r_[0, np.cumsum(ds)] if len(ds) else np.zeros(len(xyz))
        tangent = np.gradient(xyz, s_m, axis=0) if len(ds) and np.all(np.diff(s_m) > 0) else np.full_like(xyz, np.nan)
        curvature = np.linalg.norm(np.gradient(tangent, s_m, axis=0), axis=1) if np.isfinite(tangent).all() else np.full(len(xyz), np.nan)
        output.append(QualifiedPathSegment(segment.segment_id or f"{segment.source_id}:{child}", segment.source_id, segment.run_id if segment.segment_id else child, segment.source_indices or segment.input_range, _ro(xyz), _ro(s_m), _ro(tangent), _ro(curvature), uncertainty_status, status, frame or "", source_units or "", source_time if source_time is not None else float("nan"), domain or "", tuple(codes)))
    return QualifiedPathResult("1.1", tuple(output))
