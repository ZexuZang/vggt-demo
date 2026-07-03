#!/usr/bin/env python3
"""
Analyze per-frame static-weight maps exported from VGGT + Any4D + CoTracker.

This script is useful before running DROID-W:
  - mean static weight per frame
  - dynamic ratio per frame under a threshold
  - temporal smoothness of the weight maps
  - coverage plots

Example:
  python scripts/eval/static_weight_stats.py \
    --weight_dir outputs/droidw_ours_weights/person_tracking/static_weight_npy \
    --output_dir outputs_compare/eval/static_weight_stats \
    --dynamic_thresh 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install pandas: pip install pandas") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install matplotlib: pip install matplotlib") from exc


def sorted_weight_files(weight_dir: Path) -> List[Path]:
    files = sorted(list(weight_dir.glob("*.npy")))
    if not files:
        files = sorted(list(weight_dir.glob("*.png")) + list(weight_dir.glob("*.jpg")))
    if not files:
        raise FileNotFoundError(f"No .npy/.png/.jpg weight files found in {weight_dir}")
    return files


def load_weight(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Reading image weights requires opencv-python") from exc
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(path)
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dynamic_thresh", type=float, default=0.5)
    args = parser.parse_args()

    files = sorted_weight_files(args.weight_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev = None
    for i, path in enumerate(files):
        w = load_weight(path)
        static_ratio = float(np.mean(w >= args.dynamic_thresh))
        dynamic_ratio = float(np.mean(w < args.dynamic_thresh))
        mean_w = float(np.mean(w))
        median_w = float(np.median(w))
        min_w = float(np.min(w))
        max_w = float(np.max(w))
        temporal_l1 = float(np.mean(np.abs(w - prev))) if prev is not None and prev.shape == w.shape else float("nan")
        prev = w

        rows.append(
            {
                "frame_index": i,
                "file": str(path),
                "height": int(w.shape[0]),
                "width": int(w.shape[1]),
                "mean_static_weight": mean_w,
                "median_static_weight": median_w,
                "min_static_weight": min_w,
                "max_static_weight": max_w,
                "static_ratio": static_ratio,
                "dynamic_ratio": dynamic_ratio,
                "temporal_l1_to_previous": temporal_l1,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "static_weight_stats.csv", index=False)
    (args.output_dir / "static_weight_stats.json").write_text(json.dumps(rows, indent=2))

    plt.figure(figsize=(8, 4))
    plt.plot(df["frame_index"], df["mean_static_weight"], label="Mean static weight")
    plt.plot(df["frame_index"], df["dynamic_ratio"], label=f"Dynamic ratio (< {args.dynamic_thresh})")
    plt.grid(True)
    plt.xlabel("Frame index")
    plt.ylabel("Value")
    plt.title("Static-weight statistics over frames")
    plt.legend()
    plt.savefig(args.output_dir / "static_weight_stats_over_frames.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["frame_index"], df["temporal_l1_to_previous"])
    plt.grid(True)
    plt.xlabel("Frame index")
    plt.ylabel("Mean |W_t - W_{t-1}|")
    plt.title("Temporal change of static-weight maps")
    plt.savefig(args.output_dir / "static_weight_temporal_change.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(df.to_string(index=False))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
