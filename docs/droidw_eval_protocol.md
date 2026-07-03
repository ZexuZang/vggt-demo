# DROID-W + VGGT/Any4D/CoTracker Evaluation Protocol

This document summarizes the metrics used to evaluate the original DROID-W baseline and the proposed DROID-W + motion-aware static-weight method.

## 1. Camera tracking metrics

These are the most important metrics for SLAM stability.

### ATE RMSE

Absolute Trajectory Error measures the global difference between the estimated camera trajectory and the ground-truth trajectory after trajectory alignment.

Lower is better.

Report:
- ATE RMSE
- ATE mean
- ATE median
- ATE max

### RPE translation

Relative Pose Error over consecutive frames measures local frame-to-frame translational stability.

Lower is better.

### RPE rotation

Relative Pose Error over consecutive frames measures local rotational stability.

Lower is better.

## 2. Reconstruction quality metrics

Use these only when a ground-truth 3D model or ground-truth point cloud is available.

### Chamfer distance

Measures nearest-neighbor distance between predicted and ground-truth point clouds in both directions.

- Accuracy: predicted point to nearest GT point
- Completeness: GT point to nearest predicted point
- Chamfer-L1: accuracy + completeness
- Chamfer-L2: squared-distance version

Lower is better.

### F-score

Measures whether predicted and GT points match within a distance threshold.

Report F-score at several thresholds, for example:
- F-score@1cm
- F-score@2cm
- F-score@5cm

Higher is better.

### Depth metrics

For RGB-D datasets, if predicted depth or rendered depth is available:
- AbsRel
- RMSE depth
- delta accuracy
- valid depth ratio

Lower is better for errors; higher is better for accuracy and valid ratio.

## 3. Rendering / map quality metrics

If the method outputs rendered RGB images, use:
- PSNR
- SSIM
- LPIPS

These are useful for Gaussian-SLAM style mapping quality.

## 4. Dynamic/static separation metrics

If dynamic-object ground truth masks are available:
- Dynamic IoU
- Static IoU
- Precision / recall / F1 for dynamic points
- False dynamic ratio on static background
- Dynamic keep/drop ratio

If no mask GT is available, report proxy statistics:
- mean static weight
- dynamic ratio
- temporal stability of static-weight maps

## 5. Runtime metrics

Report:
- tracking FPS
- full-system FPS
- metric-depth runtime
- DINO feature extraction runtime
- final global BA runtime

DROID-W writes these to `timer_summary.csv`.

## 6. Recommended main comparison table

| Method | ATE RMSE ↓ | ATE Mean ↓ | RPE Trans RMSE ↓ | RPE Rot RMSE ↓ | Tracking FPS ↑ | Full FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DROID-W baseline | | | | | | |
| DROID-W + Ours | | | | | | |

## 7. Recommended ablation table

| Variant | Description |
|---|---|
| DROID-W baseline | original uncertainty-aware BA only |
| Ours hard | dynamic regions receive near-zero static weight |
| Ours soft | dynamic regions are down-weighted but not removed |
| Ours w/o Any4D | CoTracker + background residual only |
| Ours w/o temporal keep-alive | no multi-frame dynamic memory |
| Ours w/o depth-CC | no VGGT depth-aware connected component filtering |

## 8. Interpretation rule

The proposed method is not optimized by BA directly. It provides an external motion-aware static-weight prior. DROID-W BA still optimizes the camera trajectory and map variables, but residuals from motion-inconsistent regions receive lower weight.
