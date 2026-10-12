"""Read-only USD scene checks used after preview authoring and reopen."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SceneValidationResult:
    status:str; reason_codes:tuple; checked_prim_paths:tuple; point_counts:tuple; coordinate_error_metrics:dict; stage_metadata:dict; execution_authorized:bool=False

def validate_preview_scene(stage, artifact, *, tolerance=2e-6):
    paths=("/World/JetsonPreview","/World/JetsonPreview/LeftBoundary","/World/JetsonPreview/RightBoundary","/World/JetsonPreview/VerifiedCenterline")
    reasons=[]; counts=[]; errors={}
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
        return SceneValidationResult("FAIL",("INVALID_TOLERANCE",),paths,(),{}, {},False)
    try:
        expected_sets=(("LeftBoundary",artifact.left_boundary_xy),("RightBoundary",artifact.right_boundary_xy),("VerifiedCenterline",artifact.centerline_xy))
        for _, expected in expected_sets:
            if expected is None or len(expected)<2 or any(len(point)!=2 or not all(math.isfinite(float(v)) for v in point) for point in expected):
                return SceneValidationResult("FAIL",("EXPECTED_GEOMETRY_INVALID",),paths,(),{}, {},False)
        expected_z=float(artifact.display_transform["z_preview_m"])
        if not math.isfinite(expected_z): return SceneValidationResult("FAIL",("DISPLAY_Z_INVALID",),paths,(),{}, {},False)
    except (AttributeError,KeyError,TypeError,ValueError):
        return SceneValidationResult("FAIL",("ARTIFACT_INVALID",),paths,(),{}, {},False)
    try:
        from pxr import UsdGeom
    except ImportError as exc: raise RuntimeError("ISAAC_RUNTIME_NOT_VERIFIED") from exc
    if stage is None or UsdGeom.GetStageUpAxis(stage)!=UsdGeom.Tokens.z: reasons.append("USD_UP_AXIS_INVALID")
    meters=float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters) or abs(meters-1.)>1e-12: reasons.append("USD_METERS_PER_UNIT_INVALID")
    root=stage.GetPrimAtPath(paths[0])
    if not root.IsValid() or root.GetTypeName()!="Xform": reasons.append("PREVIEW_ROOT_INVALID")
    for name, expected in (("LeftBoundary",artifact.left_boundary_xy),("RightBoundary",artifact.right_boundary_xy),("VerifiedCenterline",artifact.centerline_xy)):
        prim=stage.GetPrimAtPath(f"/World/JetsonPreview/{name}")
        if not prim.IsValid() or prim.GetTypeName()!="BasisCurves": reasons.append(f"{name.upper()}_INVALID"); continue
        curve=UsdGeom.BasisCurves(prim); points=list(curve.GetPointsAttr().Get() or []); vertices=list(curve.GetCurveVertexCountsAttr().Get() or [])
        counts.append((name,len(points)))
        if curve.GetTypeAttr().Get()!=UsdGeom.Tokens.linear or len(vertices)!=1 or vertices[0]!=len(points) or len(points)<2: reasons.append(f"{name.upper()}_TOPOLOGY_INVALID"); continue
        actual=[(float(p[0]),float(p[1]),float(p[2])) for p in points]
        if not all(math.isfinite(v) for p in actual for v in p): reasons.append(f"{name.upper()}_NONFINITE")
        if len(actual)!=len(expected): reasons.append(f"{name.upper()}_POINT_COUNT_MISMATCH"); continue
        error=max(max(abs(x-a),abs(y-b),abs(z-expected_z)) for (x,y,z),(a,b) in zip(actual,expected))
        errors[name]=error
        if error>tolerance: reasons.append(f"{name.upper()}_COORDINATE_MISMATCH")
        widths=list(curve.GetWidthsAttr().Get() or []); colors=list(curve.GetDisplayColorAttr().Get() or [])
        if curve.GetWidthsInterpolation()!=UsdGeom.Tokens.constant or len(widths)!=1 or not math.isfinite(float(widths[0])) or widths[0]<=0: reasons.append(f"{name.upper()}_WIDTH_INVALID")
        if curve.GetDisplayColorPrimvar().GetInterpolation()!=UsdGeom.Tokens.constant or len(colors)!=1 or len(colors[0])!=3 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in colors[0]): reasons.append(f"{name.upper()}_COLOR_INVALID")
    return SceneValidationResult("PASS" if not reasons else "FAIL",tuple(reasons),paths,tuple(counts),errors,{"up_axis":"Z","meters_per_unit":meters},False)
