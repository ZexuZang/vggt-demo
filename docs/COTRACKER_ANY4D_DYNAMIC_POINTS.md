# CoTracker + Any4D Dynamic Point Mask

## Motivation

The RGB edge-aware refinement module may fail when color boundaries are weak.
This module avoids RGB edge segmentation. Instead, it uses CoTracker to track
representative 2D points and uses Any4D scene flow to decide whether those
representative points are dynamic.

## Principle

```text
Input frames
   ↓
CoTracker tracks representative 2D points
   ↓
Sample Any4D scene-flow magnitude at each representative point
   ↓
High scene-flow score + visible track = dynamic representative point
   ↓
Rasterize dynamic representative points into a dynamic probability map
   ↓
Update static BA mask / dynamic mask
```

The key idea is:

> CoTracker gives stable temporal correspondences, while Any4D gives 4D scene-flow motion. A point is treated as dynamic if its representative track has strong Any4D scene-flow magnitude.

## Colab command

Run this after generating:

```text
outputs/any4d_pred.npz
outputs/refined_mask/dynamic_probability.npy
outputs/refined_mask/static_weight.npy
```

Command:

```bash
%cd /content/vggt-demo

!python scripts/any4d_mask/cotracker_any4d_dynamic_points.py \
  --image_dir "/content/drive/MyDrive/custom4d_dataset/custom4d_unzipped/images/any4d_baby" \
  --any4d_npz outputs/any4d_pred.npz \
  --dynamic_prior_path outputs/refined_mask/dynamic_probability.npy \
  --static_weight_path outputs/refined_mask/static_weight.npy \
  --output_dir outputs/cotracker_any4d_points \
  --max_frames 8 \
  --grid_size 35 \
  --resize_max_side 768 \
  --point_dynamic_thresh 0.55 \
  --min_visible_ratio 0.35 \
  --point_radius 18 \
  --blur_sigma 7 \
  --mask_thresh 0.35 \
  --combine_with_prior max
```

## Outputs

```text
outputs/cotracker_any4d_points/
  cotracker_any4d_dynamic_probability.npy
  cotracker_any4d_dynamic_probability.png
  cotracker_any4d_dynamic_mask.png
  cotracker_any4d_dynamic_overlay.png
  cotracker_any4d_static_weight.npy
  cotracker_any4d_static_weight.png
  cotracker_any4d_static_ba_mask.png
  cotracker_any4d_static_ba_overlay.png
  representative_points_overlay.png
  dynamic_tracks_overlay.png
  cotracker_any4d_tracks.npz
  cotracker_any4d_summary.json
```

## Parameter tips

If the mask is too sparse:

```bash
--grid_size 45 --point_radius 24 --mask_thresh 0.25
```

If the mask is too large:

```bash
--point_dynamic_thresh 0.65 --mask_thresh 0.45 --combine_with_prior none
```

If Colab runs out of memory:

```bash
--grid_size 25 --resize_max_side 512 --max_frames 4
```

If you want the result to rely only on CoTracker + Any4D point scene flow:

```bash
--combine_with_prior none
```

If you want a more conservative version guided by the previous Any4D/VGGT prior:

```bash
--combine_with_prior max
```

## Difference from RGB edge-aware refinement

This module does not use GrabCut, RGB boundary fitting, or SAM. It uses RGB frames only because CoTracker needs images to track points. Dynamic/static separation is determined by Any4D scene-flow values sampled at CoTracker representative points.
