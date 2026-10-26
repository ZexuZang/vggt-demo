# SOM-Fast4D — Stage 2: Real Pre-Ours Perception

Stage 2 validates **real saved Pre-Ours outputs** in the unified Stage 1 `Fast4DPipeline`.
The real Tracking/Mapping/Prediction networks are **not integrated yet**. Their
Stage 1 mocks stay in use so perception failures can be isolated.

## Copying this patch to the existing project

Unzip this package **at the root of** `vggt-demo` on branch
`experiments/som-fast4d-integration-v1`. Merge `som_fast4d/` and `tests/` folders.
Do not delete or overwrite Stage 1 files.

Requirements for the CPU adapters: `numpy pillow`. For edge refinement also
install `opencv-python` into a compatible environment. VGGT/Any4D inference
has its *own* heavyweight dependencies and typically requires an NVIDIA GPU.

```bash
python -m pip install numpy pillow
python -m unittest discover -s tests -v
```

Expected: `Ran 15 tests` and `OK`, including 8 old tests + 7 Stage 2 tests.
This only checks contracts and synthetic test fixtures, **not neural-network accuracy**.

## Step A — Choose 6 aligned real RGB-D frames

Supply your existing Bonn `rgbd_bonn_person_tracking` directory containing:

```
DATASET/color/*.png
DATASET/depth/*.png
```

From the root of your project, run (replace DATASET with the real path):

```bash
python -m som_fast4d.make_stage2_clip \
  --dataset third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking \
  --out data/stage2_bonn_6 --start 0 --frames 6
```

This preserves original data and creates `data/stage2_bonn_6/color/000000.png ...`
and matching `depth/` with unchanged PNG depth values. If your dataset is
elsewhere, use its actual folder path. A missing dataset is a blocker; cloning
Flow4DGS-SLAM alone does not necessarily download the Bonn sequences.

## Step B — Compute real VGGT outputs on the short clip (VGGT GPU env)

```bash
python scripts/any4d_mask/export_vggt_predictions.py \
  --image_dir data/stage2_bonn_6/color \
  --output_npz outputs/stage2/vggt_6.npz --max_frames 6 --device cuda
```

The existing repo's VGGT export treats all 6 frames jointly. **This is OFFLINE
and not causally streaming.** Do not claim realtime or future-free behavior.
If CUDA memory is insufficient, use fewer frames (>=2) or a suitable GPU.

## Step C — Compute actual Any4D motion / Pre-Ours masks (Any4D GPU env)

Use the installed Any4D environment and an existing trusted checkpoint.
The checkpoint below is deliberately a placeholder you must replace:

```bash
python -m som_fast4d.run_preours_perframe \
  --image-dir data/stage2_bonn_6/color \
  --vggt-npz outputs/stage2/vggt_6.npz \
  --any4d-root third_party/Any4D \
  --any4d-checkpoint /ABSOLUTE/PATH/TO/ANY4D_CHECKPOINT \
  --any4d-python /ABSOLUTE/PATH/TO/ANY4D_PYTHON \
  --refine-python python \
  --out outputs/stage2/preours \
  --start-index 0 --end-index 6 --mode next
```

`--mode next` is explicitly **NON-CAUSAL** because current-frame masks can use
the next real image, and the VGGT output already used all 6 frames. `--mode
past` changes the Any4D target direction but does not remove the whole-clip
VGGT look-ahead. `--mode past` cannot produce frame-0 directly.

Expected per-frame files:

```
outputs/stage2/preours/000000/refined_dynamic_mask.png
outputs/stage2/preours/000000/static_weight.npy
outputs/stage2/preours/000000/static_ba_mask.png
outputs/stage2/preours/000000/dynamic_probability.npy
```

Repeat for frames 000001 ... 000005.

## Step D — Optional but recommended: original Pre-Ours edge refinement

Execute with Python containing `opencv-python`:

```bash
python -m som_fast4d.run_edge_refinement \
  --rgb-dir data/stage2_bonn_6/color \
  --preours-dir outputs/stage2/preours --max-frames 6
```

Produces `edge_refined_dynamic_mask.png`,
`edge_refined_static_weight.npy`, and
`edge_refined_static_ba_mask.npy`. Edge refinement may fail when a frame has no
foreground seed; inspect its output and adjust the original Pre-Ours thresholds,
rather than inventing a foreground object. For preliminary partial validation,
`--allow-fallback` skips failed frames (but always disclose the fallback).

## Step E — Run REAL output -> Stage 1 Pipeline (CPU)

The sample intrinsic values below are those in Flow4DGS-SLAM's Bonn example
config (`configs/rgbd/bonn/config.yaml`). Reconfirm them for your dataset.
`depth-scale=5000` means raw PNG depth value / 5000 = meters.

```bash
python -m som_fast4d.run_stage2_preours \
  --dataset data/stage2_bonn_6 \
  --preours-dir outputs/stage2/preours \
  --fx 542.822841 --fy 542.57687 \
  --cx 315.59352 --cy 237.756098 \
  --depth-scale 5000 --fps 30 --max-frames 6 \
  --resize-masks --out outputs/stage2/validation
```

`--resize-masks` is explicit because Pre-Ours outputs may have a different
resolution from 640x480 Bonn RGB-D frames. Binary masks use nearest-neighbor;
confidence uses bilinear. Always review overlays after resampling.

**Expected console structure, only if all real predictions exist and are valid:**

```
Stage 2: REAL Pre-Ours masks + MOCK tracking/mapping/prediction
frame=000000.png dynamic=... static_ba=... tracking=...
...
PASS: Pre-Ours real outputs match Stage-1 PerceptionOutput, 6 frames
```

Output files:

```
outputs/stage2/validation/000000_overlay.png
...
outputs/stage2/validation/report.json
```

Red overlay = dynamic regions. Green overlay = static BA/tracking regions.
Review frame matching, edge quality and source-camera-coordinate conventions.
Dynamic ratio alone is not proof of segmentation accuracy; evaluate masks against
annotations when available.

## Stage 2 acceptance checklist

- [ ] 15 unit tests pass in a clean environment.
- [ ] 6 RGB/depth images are aligned; original 16-bit depth is preserved.
- [ ] Pre-Ours ran on real RGB-D images (no synthetic mask fixtures).
- [ ] Each frame has dynamic mask, static confidence and static BA mask.
- [ ] `run_stage2_preours` returns PASS, saves 6 overlays and report.json.
- [ ] Manually inspect no mask inversion and reasonable overlays.
- [ ] Clearly label results OFFLINE / potentially NON-CAUSAL until future-frame
      access and compute latency are redesigned/audited.

## Commit after validation

```bash
git add som_fast4d/perception som_fast4d/make_stage2_clip.py \
  som_fast4d/run_edge_refinement.py som_fast4d/run_stage2_preours.py \
  tests/test_stage2_preours.py README_STAGE2.md

git commit -m "Add real Pre-Ours perception adapter and Stage 2 validation"
git push -u origin HEAD
```

Do not `git add .` — heavy datasets, weights, outputs and nested third-party Git
repositories should remain outside git history.

## Beyond Stage 2

Stage 3 wires the validated Pre-Ours masks into Flow4DGS-SLAM's actual
camera tracking and dynamic Gaussian mapping and runs a GPU baseline.
Stage 4 connects GenFlow3D future prediction to dynamic Gaussian initialization.
