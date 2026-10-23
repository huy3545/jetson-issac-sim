"""Export non-destructive OpenUSD preview overlay for Isaac Sim scenes.

Adheres strictly to the canonical Prim paths in configs/isaac_scene_manifest.json.
Does NOT modify or overwrite the original Isaac Sim scene assets.
"""
import argparse
import json
from pathlib import Path
import sys

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tempfile
import numpy as np

from integration.isaac_preview_adapter import (
    CANONICAL_LEFT,
    CANONICAL_RIGHT,
    IsaacPreviewEvidenceAdapter,
)
from integration.six_module_dry_run import SixModuleDryRunHarness
from simulation.isaac_manifest import IsaacSceneManifest, load_manifest
from simulation.isaac_preview_renderer import display_xyz
from simulation.isaac_scene_validator import validate_preview_scene
from vision.contracts import (
    LineVisionResult,
    ObservationStatus,
    OrientationSource,
    PointDerivationMethod,
    TrackingState,
    VisionPointProvenance,
    VisionResult,
)


def _line(name: str, points: np.ndarray) -> LineVisionResult:
    provenance = tuple(
        VisionPointProvenance(
            name,
            i,
            tuple(p),
            ObservationStatus.OBSERVED,
            TrackingState.TRACKING,
            PointDerivationMethod.DIRECT,
            lineage_available=True,
            timestamp=0.0,
            clock_domain="stage8_overlay",
        )
        for i, p in enumerate(points)
    )
    return LineVisionResult(
        points,
        ObservationStatus.OBSERVED,
        TrackingState.TRACKING,
        0.9,
        depth_valid=np.ones(len(points), bool),
        orientation_source=OrientationSource.REFERENCE,
        verified=True,
        point_provenance=provenance,
    )


def canonical_vision() -> VisionResult:
    return VisionResult(
        0.0,
        "stage8_overlay",
        frame_id="stage8-fabric-feed-miter",
        coordinate_frame="workpiece_m",
        units="m",
        left=_line("left", CANONICAL_LEFT),
        right=_line("right", CANONICAL_RIGHT),
    )


def build_overlay_stage(artifact, manifest: IsaacSceneManifest, stage):
    """Render preview curves strictly under the manifest-defined preview namespace."""
    from pxr import Gf, UsdGeom

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    namespace = manifest.preview_namespace
    root = UsdGeom.Xform.Define(stage, namespace)

    curve_specs = (
        (manifest.preview_prims["left_boundary"], artifact.left_boundary_xy, (0.0, 1.0, 0.0)),
        (manifest.preview_prims["right_boundary"], artifact.right_boundary_xy, (1.0, 0.0, 0.0)),
        (manifest.preview_prims["verified_centerline"], artifact.centerline_xy, (1.0, 1.0, 0.0)),
    )

    for prim_path, points, color in curve_specs:
        curve = UsdGeom.BasisCurves.Define(stage, prim_path)
        xyz = display_xyz(points, artifact.display_transform)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreatePointsAttr([Gf.Vec3f(*pt) for pt in xyz])
        curve.CreateCurveVertexCountsAttr([len(xyz)])
        curve.CreateWidthsAttr([0.002])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([color])
        curve.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.constant)

    return root.GetPath().pathString


def compose_sublayer_scene(target_scene_path: Path, overlay_usda_path: Path, output_composed_path: Path) -> bool:
    """Create a new non-destructive composed USD scene containing the target scene and overlay."""
    try:
        from pxr import Usd
    except ImportError:
        return False

    composed_stage = Usd.Stage.CreateNew(str(output_composed_path))
    root_layer = composed_stage.GetRootLayer()
    # Add target scene and overlay as sublayers (target first, overlay on top)
    root_layer.subLayerPaths.append(str(target_scene_path.resolve()))
    root_layer.subLayerPaths.append(str(overlay_usda_path.resolve()))
    composed_stage.Save()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export Isaac Sim non-destructive preview overlay")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to isaac_scene_manifest.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preview/jetson_overlay.usda"),
        help="Output path for standalone overlay layer",
    )
    parser.add_argument(
        "--target-scene",
        type=Path,
        default=None,
        help="Optional path to existing scene for non-destructive composition",
    )
    parser.add_argument(
        "--composed-output",
        type=Path,
        default=None,
        help="Optional path for new composed scene file (target + overlay)",
    )
    parser.add_argument("--report", type=Path, default=None, help="Output path for JSON validation report")

    args = parser.parse_args(argv)

    try:
        from pxr import Usd
    except ImportError:
        print("ISAAC_RUNTIME_NOT_VERIFIED: pxr/USD Python API is unavailable")
        return 2

    manifest = load_manifest(args.manifest)
    vision = canonical_vision()
    trace = SixModuleDryRunHarness().run(vision, run_id="stage8-preview", correlation_id="stage8-preview")
    artifact = IsaacPreviewEvidenceAdapter().build(vision, trace, preview_id="stage8-fabric-feed-preview")

    stage = Usd.Stage.CreateInMemory()
    build_overlay_stage(artifact, manifest, stage)

    scene = validate_preview_scene(stage, artifact)
    if scene.status != "PASS":
        print("ISAAC_PREVIEW_FAILED: scene validation failed")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=args.output.stem + ".",
        suffix=args.output.suffix or ".usda",
        dir=args.output.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        if not stage.Export(str(temp_path)):
            print("ISAAC_PREVIEW_FAILED: export failed")
            return 1
        temp_path.replace(args.output)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # If target scene and composed output requested, perform non-destructive sublayering
    composed_status = "SKIPPED"
    if args.target_scene and args.composed_output:
        if args.target_scene.is_file():
            args.composed_output.parent.mkdir(parents=True, exist_ok=True)
            if compose_sublayer_scene(args.target_scene, args.output, args.composed_output):
                composed_status = "PASS"
            else:
                composed_status = "FAILED"
        else:
            composed_status = f"TARGET_NOT_FOUND: {args.target_scene}"

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "status": "PASS",
            "manifest_schema": manifest.schema_version,
            "target_scene": manifest.default_stage_file,
            "preview_id": artifact.preview_id,
            "run_id": artifact.run_id,
            "scene_validation": scene.status,
            "overlay_output": str(args.output),
            "composed_status": composed_status,
            "execution_authorized": False,
            "gate_b": "BLOCKED",
        }
        args.report.write_text(json.dumps(report_data, indent=2) + "\n")

    print(f"ISAAC_OVERLAY_SUCCESS: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
