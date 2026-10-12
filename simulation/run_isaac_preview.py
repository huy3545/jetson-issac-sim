"""Stage 7D-2C CLI for a disposable, read-only Isaac/USD preview."""
import argparse
from pathlib import Path
import sys
import numpy as np
from integration.isaac_preview_adapter import CANONICAL_LEFT, CANONICAL_RIGHT, IsaacPreviewEvidenceAdapter
from integration.six_module_dry_run import SixModuleDryRunHarness
from simulation.isaac_preview_renderer import render_preview
from vision.contracts import (LineVisionResult, ObservationStatus, OrientationSource,
                              PointDerivationMethod, TrackingState, VisionPointProvenance, VisionResult)

def _line(name, points):
    provenance=tuple(VisionPointProvenance(name,i,tuple(p),ObservationStatus.OBSERVED,TrackingState.TRACKING,PointDerivationMethod.DIRECT,lineage_available=True,timestamp=0.,clock_domain="stage7d2c") for i,p in enumerate(points))
    return LineVisionResult(points,ObservationStatus.OBSERVED,TrackingState.TRACKING,.9,depth_valid=np.ones(len(points),bool),orientation_source=OrientationSource.REFERENCE,verified=True,point_provenance=provenance)

def canonical_vision():
    return VisionResult(0.,"stage7d2c",frame_id="stage7d2c-square-miter-90",coordinate_frame="workpiece_m",units="m",left=_line("left",CANONICAL_LEFT),right=_line("right",CANONICAL_RIGHT))

def main(argv=None):
    parser=argparse.ArgumentParser(description="Read-only Stage 7D-2C Isaac preview")
    parser.add_argument("--fixture",default="square-miter-90",choices=("square-miter-90",))
    parser.add_argument("--output",type=Path,default=Path("artifacts/preview/stage7d2c_preview.usda"))
    args=parser.parse_args(argv)
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("ISAAC_RUNTIME_NOT_VERIFIED: pxr/USD Python API is unavailable")
        return 2
    vision=canonical_vision(); trace=SixModuleDryRunHarness().run(vision,run_id="stage7d2c-preview",correlation_id="stage7d2c-preview")
    artifact=IsaacPreviewEvidenceAdapter().build(vision,trace,preview_id="stage7d2c-square-miter-90")
    stage=Usd.Stage.CreateInMemory(); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); stage.SetMetersPerUnit(1.0)
    root=render_preview(artifact,vision,trace,stage)
    expected=(root,"/World/JetsonPreview/LeftBoundary","/World/JetsonPreview/RightBoundary","/World/JetsonPreview/VerifiedCenterline")
    if not all(stage.GetPrimAtPath(path).IsValid() for path in expected):
        print("ISAAC_PREVIEW_FAILED: required preview prim missing"); return 1
    args.output.parent.mkdir(parents=True,exist_ok=True); stage.Export(str(args.output))
    print(f"ISAAC_PREVIEW_SUCCESS: {args.output}")
    return 0

if __name__=="__main__": sys.exit(main())
