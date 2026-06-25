# Pure Point-Level CoTracker + Any4D Dynamic Points

This module outputs point-level dynamic labels, not a dense mask.

## Removed smoothing steps

This version removes:

- circle rasterization into dense blobs
- Gaussian blur
- thresholded dense object mask
- RGB GrabCut
- SAM
- dynamic prior fusion

## Method

```text
Input frames
   ↓
CoTracker tracks representative 2D points
   ↓
Any4D provides scene-flow magnitude on the reference frame
   ↓
Sample scene-flow magnitude at each representative point
   ↓
Classify each tracked point as dynamic or static
   ↓
Save point-level labels, scores, and tracks
```

## Colab command

```bash
%cd /content/vggt-demo

!python scripts/any4d_mask/cotracker_any4d_point_level.py \
  --image_dir "/content/drive/MyDrive/custom4d_dataset/custom4d_unzipped/images/any4d_baby" \
  --any4d_npz outputs/any4d_pred.npz \
  --output_dir outputs/cotracker_any4d_point_level \
  --max_frames 8 \
  --grid_size 45 \
  --resize_max_side 768 \
  --point_dynamic_thresh 0.55 \
  --min_visible_ratio 0.35 \
  --point_vis_radius 3 \
  --point_mask_radius 1
```

## Outputs

```text
outputs/cotracker_any4d_point_level/
  point_level_dynamic_tracks.npz
  point_level_dynamic_points.csv
  point_level_dynamic_mask.png
  point_level_points_overlay.png
  point_level_dynamic_tracks_overlay.png
  any4d_motion_score_map_resized.png
  point_level_summary.json
```

## Key output

`point_level_dynamic_tracks.npz` contains:

```text
tracks: T x N x 2
visibility: T x N
initial_points_resized: N x 2
initial_points_original: N x 2
point_motion_score: N
point_dynamic_prob: N
dynamic_labels: N
visible_ratio: N
```

## Parameter tips

If too few dynamic points:

```bash
--point_dynamic_thresh 0.45 --grid_size 55
```

If too many dynamic points:

```bash
--point_dynamic_thresh 0.65
```

If CoTracker uses too much memory:

```bash
--grid_size 25 --resize_max_side 512 --max_frames 4
```
