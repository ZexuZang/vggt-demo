# RGB Edge-Aware Refinement Module

## Motivation

The Any4D scene-flow dynamic mask is stable and has fewer ghosting artifacts,
but it can be blob-like and lose fine boundaries such as the gap between legs
or stroller details.

This module uses RGB information to refine the dynamic mask boundary.

## Method

```text
Any4D / VGGT dynamic probability
        ↓
Convert probability to GrabCut seeds
        ↓
Run RGB-guided GrabCut
        ↓
Keep components overlapping the original motion prior
        ↓
Export edge-refined dynamic mask
        ↓
Update static BA mask / static weight
```

## Colab command

```bash
%cd /content/vggt-demo

!python scripts/any4d_mask/edge_aware_refine.py \
  --image_dir "/content/drive/MyDrive/custom4d_dataset/custom4d_unzipped/images/any4d_baby" \
  --prob_path outputs/refined_mask/dynamic_probability.npy \
  --static_weight_path outputs/refined_mask/static_weight.npy \
  --output_dir outputs/refined_mask_edge \
  --ref_idx 0 \
  --fg_thresh 0.75 \
  --pr_fg_thresh 0.35 \
  --bg_thresh 0.10 \
  --dilate_px 7 \
  --grabcut_iters 5 \
  --open_px 1 \
  --close_px 0
```

## Outputs

```text
outputs/refined_mask_edge/
  input_dynamic_probability_resized.png
  edge_refined_dynamic_mask.png
  edge_refined_dynamic_overlay.png
  edge_refined_static_weight.png
  edge_refined_static_ba_mask.png
  edge_refined_static_ba_overlay.png
  grabcut_seed_visualization.png
  edge_refine_summary.json
```

## Parameter tips

If the mask is too large:

```bash
--fg_thresh 0.80 --pr_fg_thresh 0.45
```

If the mask misses object edges:

```bash
--fg_thresh 0.65 --pr_fg_thresh 0.25 --dilate_px 11
```

If the gaps between legs are still filled too much:

```bash
--close_px 0 --open_px 0
```

If there are many small noise regions:

```bash
--min_area 800
```
