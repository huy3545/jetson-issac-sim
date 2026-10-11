"""Lazy, optional Isaac/USD renderer for a validated read-only preview."""
from integration.isaac_preview_contracts import IsaacPreviewArtifact


def display_xyz(points_xy, transform):
    """Convert metric XY to display-only XYZ without changing source metadata."""
    if transform.get("kind") != "DISPLAY_ONLY_XY_TO_XYZ" or transform.get("usd_up_axis") != "Z":
        raise ValueError("unsupported display transform")
    scale=transform.get("scale_m_per_stage_unit"); z=transform.get("z_preview_m")
    if scale != 1.0 or not isinstance(z,(int,float)): raise ValueError("unsupported display scale")
    return tuple((x,y,float(z)) for x,y in points_xy)


def render_preview(artifact: IsaacPreviewArtifact, stage):
    """Render only when a caller supplies a USD stage; no Isaac imports at module load."""
    if not artifact.preview_authorized or artifact.execution_authorized:
        raise ValueError("preview is not authorized for centerline rendering")
    try:
        from pxr import UsdGeom, Gf
    except ImportError as exc:
        raise RuntimeError("ISAAC_RUNTIME_NOT_VERIFIED") from exc
    root=UsdGeom.Xform.Define(stage, "/World/JetsonPreview")
    for name, points, color in (("LeftBoundary",artifact.left_boundary_xy,(0.,1.,0.)),("RightBoundary",artifact.right_boundary_xy,(1.,0.,0.)),("VerifiedCenterline",artifact.centerline_xy,(1.,1.,0.))):
        curve=UsdGeom.BasisCurves.Define(stage, f"/World/JetsonPreview/{name}")
        xyz=display_xyz(points,artifact.display_transform)
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in xyz]); curve.CreateCurveVertexCountsAttr([len(xyz)])
        curve.CreateDisplayColorAttr([color])
    return root.GetPath().pathString
