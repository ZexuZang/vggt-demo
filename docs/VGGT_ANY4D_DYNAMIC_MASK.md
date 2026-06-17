# VGGT + Any4D Dynamic Mask Refinement

This package adds an **inference-time experiment** on top of your existing `vggt-demo` repo.

It does **not** retrain VGGT or Any4D. It uses:

1. VGGT initial geometry:
   - camera pose
   - depth map
   - point map
   - confidence / uncertainty
2. Any4D dynamic modeling:
   - scene flow
   - 4D geometry
   - dynamic motion cue
3. Fusion:
   - dynamic probability map
   - refined dynamic mask
   - static BA mask / static weight

## Folder layout

Copy the four Python scripts into your repo:

```text
vggt-demo/
  tools/
    export_vggt_predictions.py
    export_any4d_predictions.py
    refine_dynamic_mask.py
    visualize_mask_overlay.py
```

Clone Any4D into:

```text
vggt-demo/
  third_party/
    Any4D/
```

## Install Any4D

```bash
cd vggt-demo
mkdir -p third_party
git clone https://github.com/Any-4D/Any4D.git third_party/Any4D

conda create -n any4d python=3.12 -y
conda activate any4d
cd third_party/Any4D
pip install -e ".[all]"

mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth
```

## Step 1: Export VGGT predictions

Run this in your VGGT environment:

```bash
conda activate vggt
cd /path/to/vggt-demo

python tools/export_vggt_predictions.py \
  --image_dir data/stroller_frames \
  --output_npz outputs/vggt_pred.npz
```

## Step 2: Export Any4D predictions

Run this in the Any4D environment:

```bash
conda activate any4d
cd /path/to/vggt-demo

python tools/export_any4d_predictions.py \
  --any4d_root third_party/Any4D \
  --image_dir data/stroller_frames \
  --checkpoint_path third_party/Any4D/checkpoints/any4d_4v_combined.pth \
  --output_npz outputs/any4d_pred.npz \
  --ref_img_idx 0 \
  --start_idx 0 \
  --end_idx 8
```

## Step 3: Refine dynamic mask

Run this in either environment with numpy/opencv installed:

```bash
cd /path/to/vggt-demo

python tools/refine_dynamic_mask.py \
  --vggt_npz outputs/vggt_pred.npz \
  --any4d_npz outputs/any4d_pred.npz \
  --image_dir data/stroller_frames \
  --output_dir outputs/refined_mask \
  --ref_idx 0 \
  --aggregate max \
  --dynamic_thresh 0.55 \
  --static_weight_thresh 0.45
```

## Outputs

```text
outputs/refined_mask/
  motion_probability.npy
  motion_probability.png
  vggt_confidence.npy
  vggt_confidence.png
  dynamic_probability.npy
  dynamic_probability.png
  static_weight.npy
  static_weight.png
  refined_dynamic_mask.png
  static_ba_mask.png
  exclude_from_static_ba_mask.png
  refined_dynamic_overlay.png
  static_ba_overlay.png
  summary.json
```

Use:

- `static_weight.npy`: soft weights for pose refinement / BA.
- `static_ba_mask.png`: hard static mask for SLAM / BA.
- `refined_dynamic_mask.png`: dynamic region for 4D tracking.
- `dynamic_probability.npy`: soft dynamic probability map.

## Core formula

For each reference-frame pixel, Any4D gives a scene-flow-based motion magnitude:

```text
motion_mag = ||scene_flow||
```

This is converted into a dynamic probability:

```text
P_dynamic = robust_normalize(motion_mag)
```

Then VGGT confidence is used to preserve static high-confidence regions:

```text
static_weight = (1 - P_dynamic) * VGGT_confidence
```

The static BA backend should use `static_weight`, not only a hard binary mask.

## Notes

Any4D inference code is still being updated upstream. The exporter prints all Any4D prediction keys to:

```text
outputs/any4d_pred.keys.json
```

If output key names change, adapt `export_any4d_predictions.py` using that key report.
