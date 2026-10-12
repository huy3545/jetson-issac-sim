"""Read-only USD scene checks used after preview authoring and reopen."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SceneValidationResult:
    status:str; reason_codes:tuple; checked_prim_paths:tuple; point_counts:tuple; coordinate_error_metrics:dict; stage_metadata:dict; execution_authorized:bool=False

def validate_preview_scene(stage, artifact, *, tolerance=2e-6):
    paths=("/World/JetsonPreview","/World/JetsonPreview/LeftBoundary","/World/JetsonPreview/RightBoundary","/World/JetsonPreview/VerifiedCenterline")
    try:
        from pxr import UsdGeom
    except ImportError as exc: raise RuntimeError("ISAAC_RUNTIME_NOT_VERIFIED") from exc
    reasons=[]; counts=[]; errors={}
    if stage is None or UsdGeom.GetStageUpAxis(stage)!=UsdGeom.Tokens.z: reasons.append("USD_UP_AXIS_INVALID")
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage))-1.)>1e-12: reasons.append("USD_METERS_PER_UNIT_INVALID")
    root=stage.GetPrimAtPath(paths[0])
    if not root.IsValid() or root.GetTypeName()!="Xform": reasons.append("PREVIEW_ROOT_INVALID")
    for name, expected in (("LeftBoundary",artifact.left_boundary_xy),("RightBoundary",artifact.right_boundary_xy),("VerifiedCenterline",artifact.centerline_xy)):
        prim=stage.GetPrimAtPath(f"/World/JetsonPreview/{name}")
        if not prim.IsValid() or prim.GetTypeName()!="BasisCurves": reasons.append(f"{name.upper()}_INVALID"); continue
        curve=UsdGeom.BasisCurves(prim); points=curve.GetPointsAttr().Get() or []; vertices=curve.GetCurveVertexCountsAttr().Get() or []
        counts.append((name,len(points)))
        if curve.GetTypeAttr().Get()!=UsdGeom.Tokens.linear or vertices!=[len(points)] or len(points)<2: reasons.append(f"{name.upper()}_TOPOLOGY_INVALID"); continue
        actual=[(float(p[0]),float(p[1]),float(p[2])) for p in points]
        if not all(math.isfinite(v) for p in actual for v in p): reasons.append(f"{name.upper()}_NONFINITE")
        error=max((max(abs(x-a),abs(y-b),abs(z)) for (x,y,z),(a,b) in zip(actual,expected)),default=float("inf"))
        errors[name]=error
        if len(actual)!=len(expected) or error>tolerance: reasons.append(f"{name.upper()}_COORDINATE_MISMATCH")
        widths=curve.GetWidthsAttr().Get() or []; colors=curve.GetDisplayColorAttr().Get() or []
        if not widths or not math.isfinite(float(widths[0])) or widths[0]<=0: reasons.append(f"{name.upper()}_WIDTH_INVALID")
        if not colors or len(colors[0])!=3: reasons.append(f"{name.upper()}_COLOR_INVALID")
    return SceneValidationResult("PASS" if not reasons else "FAIL",tuple(reasons),paths,tuple(counts),errors,{"up_axis":"Z","meters_per_unit":float(UsdGeom.GetStageMetersPerUnit(stage))},False)
