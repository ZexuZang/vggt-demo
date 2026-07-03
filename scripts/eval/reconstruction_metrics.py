#!/usr/bin/env python3
"""
Point-cloud reconstruction metrics.

Use this when you have a predicted reconstruction and a ground-truth point cloud.
It computes common geometric reconstruction metrics:
  - Accuracy: mean nearest-neighbor distance from predicted points to GT
  - Completeness: mean nearest-neighbor distance from GT points to prediction
  - Chamfer-L1: accuracy + completeness
  - Chamfer-L2: mean squared nearest-neighbor distances in both directions
  - Precision / Recall / F-score at one or more distance thresholds

Supported input:
  - .ply, .pcd via Open3D
  - .npy array shaped [N, 3] or [N, >=3]
  - .npz with common keys: points, xyz, pts3d, point_cloud, vertices

Example:
  python scripts/eval/reconstruction_metrics.py \
    --pred outputs/pred_points.ply \
    --gt data/gt_points.ply \
    --output_dir outputs/eval/reconstruction \
    --thresholds 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install scipy: pip install scipy") from exc


def load_points(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".npy"]:
        arr = np.load(path)
    elif suffix in [".npz"]:
        npz = np.load(path)
        for key in ["points", "xyz", "pts3d", "point_cloud", "vertices", "pred_points", "gt_points"]:
            if key in npz:
                arr = npz[key]
                break
        else:
            keys = list(npz.keys())
            raise KeyError(f"No known point key found in {path}. Available keys: {keys}")
    elif suffix in [".ply", ".pcd"]:
        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError("Reading .ply/.pcd requires open3d: pip install open3d") from exc
        pcd = o3d.io.read_point_cloud(str(path))
        arr = np.asarray(pcd.points)
    else:
        # fallback for txt/csv with xyz columns
        arr = np.loadtxt(path)

    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 3:
        raise ValueError(f"Point cloud must have at least 3 columns, got {arr.shape}")

    xyz = arr[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    if len(xyz) == 0:
        raise ValueError(f"No valid points in {path}")
    return xyz


def random_downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(idx)]


def nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1, workers=-1)
    return dist.astype(np.float64)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    pred_to_gt = nearest_distances(pred, gt)
    gt_to_pred = nearest_distances(gt, pred)

    metrics: Dict[str, float] = {
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
        "accuracy_mean": float(np.mean(pred_to_gt)),
        "accuracy_median": float(np.median(pred_to_gt)),
        "accuracy_rmse": float(np.sqrt(np.mean(pred_to_gt**2))),
        "completeness_mean": float(np.mean(gt_to_pred)),
        "completeness_median": float(np.median(gt_to_pred)),
        "completeness_rmse": float(np.sqrt(np.mean(gt_to_pred**2))),
        "chamfer_l1": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "chamfer_l2": float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)),
    }

    for tau in thresholds:
        precision = float(np.mean(pred_to_gt < tau))
        recall = float(np.mean(gt_to_pred < tau))
        fscore = 0.0 if precision + recall < 1e-12 else float(2.0 * precision * recall / (precision + recall))
        key = f"{tau:g}"
        metrics[f"precision@{key}"] = precision
        metrics[f"recall@{key}"] = recall
        metrics[f"fscore@{key}"] = fscore

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pred = load_points(args.pred)
    gt = load_points(args.gt)

    pred = random_downsample(pred, args.max_points, seed=args.seed)
    gt = random_downsample(gt, args.max_points, seed=args.seed + 1)

    metrics = compute_metrics(pred, gt, args.thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import pandas as pd
        pd.DataFrame([metrics]).to_csv(args.output_dir / "reconstruction_metrics.csv", index=False)
    except Exception:
        pass

    print(json.dumps(metrics, indent=2))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
