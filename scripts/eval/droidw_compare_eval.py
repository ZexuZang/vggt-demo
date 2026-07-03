#!/usr/bin/env python3
"""
Compare DROID-W baseline vs DROID-W + external static weights.

This script is intentionally independent from evo. It reads DROID-W output folders,
parses DROID-W's own ATE metric files when available, and also recomputes:
  - ATE after Sim(3) Umeyama alignment
  - RPE translation over consecutive frame pairs
  - RPE rotation over consecutive frame pairs
  - trajectory and error plots

Expected DROID-W output folder:
  <scene_dir>/gt_poses.txt
  <scene_dir>/traj/est_poses_full.txt
  <scene_dir>/traj/metrics_full_traj.txt
  <scene_dir>/timer_summary.csv

Example:
  python scripts/eval/droidw_compare_eval.py \
    --baseline_dir /content/vggt-demo/outputs_compare/droidw_baseline \
    --ours_dir /content/vggt-demo/outputs_compare/droidw_ours \
    --output_dir /content/vggt-demo/outputs_compare/eval
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install pandas: pip install pandas") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install matplotlib: pip install matplotlib") from exc


def _clean_np_float_repr(text: str) -> str:
    # DROID-W sometimes writes: np.float64(0.123)
    text = re.sub(r"np\.float\d+\(([^()]+)\)", r"\1", text)
    return text


def parse_droidw_metric_file(path: Path) -> Optional[Dict[str, float]]:
    """Parse DROID-W metrics_*.txt statistics dict."""
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"statistics:\s*(\{.*?\})", text, re.S)
    if not match:
        return None
    payload = _clean_np_float_repr(match.group(1))
    data = ast.literal_eval(payload)
    return {k: float(v) for k, v in data.items()}


def load_tum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TUM trajectory format:
        timestamp tx ty tz qx qy qz qw
    Returns timestamps, xyz, quaternion qx qy qz qw.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 8:
        raise ValueError(f"Expected TUM format with >=8 columns, got {data.shape} in {path}")

    timestamps = data[:, 0].astype(np.float64)
    xyz = data[:, 1:4].astype(np.float64)
    quat = data[:, 4:8].astype(np.float64)
    return timestamps, xyz, quat


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion qx,qy,qz,qw -> 3x3 rotation matrix."""
    qx, qy, qz, qw = [float(x) for x in q]
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_angle_deg(R: np.ndarray) -> float:
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def umeyama_align(
    X: np.ndarray,
    Y: np.ndarray,
    with_scale: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align estimated points X to reference Y using Sim(3) Umeyama alignment.
    Returns aligned X, scale, rotation, translation.
    """
    n = min(len(X), len(Y))
    X = X[:n].astype(np.float64)
    Y = Y[:n].astype(np.float64)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    if with_scale:
        var_x = float((Xc**2).sum() / n)
        scale = float(np.trace(np.diag(D) @ S) / max(var_x, 1e-12))
    else:
        scale = 1.0

    t = mu_y - scale * R @ mu_x
    X_aligned = scale * (X @ R.T) + t
    return X_aligned, scale, R, t


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan"),
        "mean": float(np.mean(errors)) if len(errors) else float("nan"),
        "median": float(np.median(errors)) if len(errors) else float("nan"),
        "std": float(np.std(errors)) if len(errors) else float("nan"),
        "min": float(np.min(errors)) if len(errors) else float("nan"),
        "max": float(np.max(errors)) if len(errors) else float("nan"),
        "sse": float(np.sum(errors**2)) if len(errors) else float("nan"),
    }


def evaluate_trajectory(scene_dir: Path) -> Dict[str, Any]:
    gt_path = scene_dir / "gt_poses.txt"
    est_path = scene_dir / "traj" / "est_poses_full.txt"

    gt_t, gt_xyz, gt_q = load_tum(gt_path)
    est_t, est_xyz, est_q = load_tum(est_path)

    # DROID-W sometimes writes GT with original timestamps and est with frame indices.
    # For a single sequence run, index matching is more robust.
    n = min(len(gt_xyz), len(est_xyz), len(gt_q), len(est_q))
    gt_xyz = gt_xyz[:n]
    est_xyz = est_xyz[:n]
    gt_q = gt_q[:n]
    est_q = est_q[:n]

    est_aligned, scale, R_align, trans = umeyama_align(est_xyz, gt_xyz, with_scale=True)

    ate_errors = np.linalg.norm(est_aligned - gt_xyz, axis=1)
    ate = summarize_errors(ate_errors)

    rpe_trans_errors: List[float] = []
    rpe_rot_errors: List[float] = []

    for i in range(n - 1):
        gt_dt = gt_xyz[i + 1] - gt_xyz[i]
        est_dt = est_aligned[i + 1] - est_aligned[i]
        rpe_trans_errors.append(float(np.linalg.norm(est_dt - gt_dt)))

        Rg0 = quat_to_rot(gt_q[i])
        Rg1 = quat_to_rot(gt_q[i + 1])
        Re0 = quat_to_rot(est_q[i])
        Re1 = quat_to_rot(est_q[i + 1])

        Rg_rel = Rg0.T @ Rg1
        Re_rel = Re0.T @ Re1
        R_err = Rg_rel.T @ Re_rel
        rpe_rot_errors.append(rot_angle_deg(R_err))

    rpe_trans = summarize_errors(np.array(rpe_trans_errors))
    rpe_rot = summarize_errors(np.array(rpe_rot_errors))

    droid_full = parse_droidw_metric_file(scene_dir / "traj" / "metrics_full_traj.txt")
    droid_kf = parse_droidw_metric_file(scene_dir / "traj" / "metrics_kf_traj.txt")
    droid_kf_before_ba = parse_droidw_metric_file(
        scene_dir / "traj" / "before_final_ba" / "metrics_kf_traj.txt"
    )

    return {
        "frames": int(n),
        "ate": ate,
        "rpe_trans": rpe_trans,
        "rpe_rot": rpe_rot,
        "scale": float(scale),
        "alignment_rotation": R_align.tolist(),
        "alignment_translation": trans.tolist(),
        "gt_xyz": gt_xyz,
        "est_xyz": est_xyz,
        "est_aligned": est_aligned,
        "ate_errors": ate_errors,
        "rpe_trans_errors": np.array(rpe_trans_errors),
        "rpe_rot_errors": np.array(rpe_rot_errors),
        "droidw_full": droid_full,
        "droidw_keyframe": droid_kf,
        "droidw_keyframe_before_final_ba": droid_kf_before_ba,
    }


def percent_improvement(base: float, ours: float) -> float:
    if not np.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return float((base - ours) / base * 100.0)


def make_comparison_tables(
    baseline: Dict[str, Any],
    ours: Dict[str, Any],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, res in [("DROID-W baseline", baseline), ("DROID-W + Ours", ours)]:
        rows.append(
            {
                "Method": method,
                "Frames": res["frames"],
                "ATE RMSE [m] ↓": res["ate"]["rmse"],
                "ATE Mean [m] ↓": res["ate"]["mean"],
                "ATE Median [m] ↓": res["ate"]["median"],
                "ATE Max [m] ↓": res["ate"]["max"],
                "RPE Trans RMSE [m] ↓": res["rpe_trans"]["rmse"],
                "RPE Trans Mean [m] ↓": res["rpe_trans"]["mean"],
                "RPE Rot RMSE [deg] ↓": res["rpe_rot"]["rmse"],
                "RPE Rot Mean [deg] ↓": res["rpe_rot"]["mean"],
                "Scale": res["scale"],
            }
        )

    metrics_df = pd.DataFrame(rows)

    imp = {
        "Metric": [],
        "Baseline": [],
        "Ours": [],
        "Absolute improvement": [],
        "Improvement [%]": [],
    }
    for col in [
        "ATE RMSE [m] ↓",
        "ATE Mean [m] ↓",
        "ATE Median [m] ↓",
        "ATE Max [m] ↓",
        "RPE Trans RMSE [m] ↓",
        "RPE Rot RMSE [deg] ↓",
    ]:
        b = float(metrics_df.loc[0, col])
        o = float(metrics_df.loc[1, col])
        imp["Metric"].append(col)
        imp["Baseline"].append(b)
        imp["Ours"].append(o)
        imp["Absolute improvement"].append(b - o)
        imp["Improvement [%]"].append(percent_improvement(b, o))
    improvement_df = pd.DataFrame(imp)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "trajectory_metrics_baseline_vs_ours.csv", index=False)
    improvement_df.to_csv(output_dir / "trajectory_improvement_baseline_vs_ours.csv", index=False)

    return metrics_df, improvement_df


def plot_trajectories(baseline: Dict[str, Any], ours: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(baseline["gt_xyz"]), len(baseline["est_aligned"]), len(ours["est_aligned"]))
    gt = baseline["gt_xyz"][:n]
    base_est = baseline["est_aligned"][:n]
    ours_est = ours["est_aligned"][:n]

    plt.figure(figsize=(7, 6))
    plt.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=2)
    plt.plot(base_est[:, 0], base_est[:, 1], label="DROID-W baseline")
    plt.plot(ours_est[:, 0], ours_est[:, 1], label="DROID-W + Ours")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Camera trajectory XY")
    plt.legend()
    plt.savefig(output_dir / "trajectory_xy_baseline_vs_ours.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(gt[:, 0], gt[:, 2], label="GT", linewidth=2)
    plt.plot(base_est[:, 0], base_est[:, 2], label="DROID-W baseline")
    plt.plot(ours_est[:, 0], ours_est[:, 2], label="DROID-W + Ours")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Camera trajectory XZ")
    plt.legend()
    plt.savefig(output_dir / "trajectory_xz_baseline_vs_ours.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_error_curves(baseline: Dict[str, Any], ours: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, ylabel, title, fname in [
        ("ate_errors", "ATE error [m]", "ATE error over frames", "ate_error_over_frames.png"),
        (
            "rpe_trans_errors",
            "RPE translation error [m]",
            "RPE translation error over frame pairs",
            "rpe_translation_error_over_frames.png",
        ),
        (
            "rpe_rot_errors",
            "RPE rotation error [deg]",
            "RPE rotation error over frame pairs",
            "rpe_rotation_error_over_frames.png",
        ),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(baseline[key], label="DROID-W baseline")
        plt.plot(ours[key], label="DROID-W + Ours")
        plt.grid(True)
        plt.xlabel("Frame index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(output_dir / fname, dpi=220, bbox_inches="tight")
        plt.close()


def save_json_summary(
    baseline: Dict[str, Any],
    ours: Dict[str, Any],
    output_dir: Path,
) -> None:
    def strip_arrays(res: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                continue
            clean[k] = v
        return clean

    summary = {
        "baseline": strip_arrays(baseline),
        "ours": strip_arrays(ours),
        "interpretation": {
            "positive_improvement_means": "Baseline metric - ours metric; positive is better for error metrics.",
        },
    }
    (output_dir / "trajectory_eval_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True, type=Path)
    parser.add_argument("--ours_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    baseline = evaluate_trajectory(args.baseline_dir)
    ours = evaluate_trajectory(args.ours_dir)

    metrics_df, improvement_df = make_comparison_tables(baseline, ours, args.output_dir)
    plot_trajectories(baseline, ours, args.output_dir)
    plot_error_curves(baseline, ours, args.output_dir)
    save_json_summary(baseline, ours, args.output_dir)

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)

    print("\n==== Trajectory metrics ====")
    print(metrics_df.to_string(index=False))
    print("\n==== Improvement, positive is better ====")
    print(improvement_df.to_string(index=False))
    print(f"\nSaved evaluation to: {args.output_dir}")


if __name__ == "__main__":
    main()
