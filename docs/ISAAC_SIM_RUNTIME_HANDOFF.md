# Isaac Sim runtime handoff

This branch requires an NVIDIA Isaac Sim Python environment with `pxr`/OpenUSD. Blender is not a substitute for this qualification.

Check out `jetson-v1.1-stage7d2-isaac-preview`, then use the Isaac Sim supplied Python launcher rather than replacing its bundled USD libraries. Install only project dependencies compatible with that environment (NumPy and OpenCV from `requirements.txt`).

Run:

```bash
<isaac-python> -m simulation.run_isaac_preview --fixture square-miter-90 --output artifacts/preview/stage7d2c_preview.usda --report artifacts/preview/stage7d2c_report.json
```

The output must contain `/World/JetsonPreview/{LeftBoundary,RightBoundary,VerifiedCenterline}` as linear `BasisCurves`, with Z-up and one stage unit per metre. Open the exported USD in Isaac Sim and inspect the three curves. A successful OpenUSD export validates only the USD Runtime Gate; actual Isaac Sim execution is required for the Isaac Sim Runtime Gate.

The preview is read-only synthetic geometry. Gate B remains BLOCKED and `execution_authorized=false`; there are no controls, cutting commands or hardware interfaces.
