# Stage 1: Unified Interfaces and Causal Pipeline (CPU only)

## What this stage does

Adds `som_fast4d/core/data_types.py`, `som_fast4d/core/pipeline.py`, and
`som_fast4d/run_stage1_smoke.py`. These components define the interface between:

- Pre-Ours dynamic mask + static confidence (eventually an adapter)
- Flow4DGS-SLAM camera tracking / mapping (eventually adapters)
- GenFlow3D-inspired history-aware next-frame scene-flow prediction (eventually an adapter)

This stage **does not execute those deep models**. The smoke test uses mock
modules to verify that the prediction at t is passed only to t+1 mapping and
that the t+1 observation can correct that predicted initialization.

## VS Code steps

Copy `som_fast4d/` and `tests/` from this overlay ZIP into the root of your
existing `vggt-demo` repository (merge folders; do not replace existing folders).
Keep the `README_STAGE1.md` at the repo root.

In VS Code integrated terminal at `vggt-demo` root:

```bash
python -m pip install numpy pillow
python -m unittest discover -s tests -v
python -m som_fast4d.run_stage1_smoke
```

On some Windows machines `python` may be invoked as `py` instead; use the same
interpreter which ran the previous adapter tests.

Expected ending:

```text
Ran 8 tests

OK
...
frame=2 ... prior_used=1 ... prediction_for_t+1=yes
frame=3 ... prior_used=1 ... prediction_for_t+1=yes
PASS: causal prediction -> next-frame measurement correction contract
```

**Important**: The tests verify only array contracts, frame ordering, mask
semantics, and the staged data flow. They do not validate camera tracking,
learned motion inference, rendering, CUDA, or real-world FPS.

## Design conventions

| Field | Convention |
| --- | --- |
| `FrameData.rgb` | `uint8 [H,W,3]`, RGB order |
| `FrameData.depth_m` | float `[H,W]`, meters |
| `PerceptionOutput.dynamic_mask` | bool `[H,W]`: `True = dynamic` |
| `PerceptionOutput.static_ba_mask` | bool `[H,W]`: `True = usable static` |
| `CameraEstimate.T_world_camera` | 4×4 pose mapping camera -> world |
| `GaussianMapState.centers_world_m` | float `[N,3]` in world coordinates, meters |
| `MotionPrediction.delta_world_m` | float `[M,3]` scene displacement in world, meters |
| `MotionPrediction.gaussian_ids` | stable Gaussian IDs, *not* array-row indices |

`MotionPrediction.uncertainty_std_m` is optional, not an already-calibrated model uncertainty estimate.

## Git upload

```bash
git status --short
git add som_fast4d/core som_fast4d/run_stage1_smoke.py tests/test_stage1_pipeline.py README_STAGE1.md
git commit -m "Add unified Fast4D data contracts and causal pipeline scaffold"
git push -u origin HEAD
```

Do not run `git add .` without checking `third_party` / checkpoint outputs.

## Next stage (GPU)

A real Pre-Ours adapter must consume a causal input window and return an
`PerceptionOutput`. A Flow4DGS-SLAM tracking adapter returns `CameraEstimate`,
then a Mapping adapter outputs world-frame centers and stable IDs. A GenFlow3D
adapter can only use the maps / clouds available up to time t and must return
a `MotionPrediction` for t+1. Correction uses real RGB-D and optical flow after
frame t+1 arrives. Actual Flow4DGS-SLAM internals include other Gaussian
attributes and optimizer state and should not be overwritten by this skeleton.
