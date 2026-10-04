#!/usr/bin/env python3
"""
Three-way trajectory evaluation for Baseline 0 / Baseline 1 / Ours.

This script reuses the trajectory parsing, Umeyama Sim(3) alignment, ATE and RPE
implementation already present in:
    scripts/eval/droidw_compare_eval.py

Expected run folders:
    <root_dir>/baseline0
    <root_dir>/baseline1
    <root_dir>/ours
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHODS = ("Baseline 0", "Baseline 1", "Ours")
FOLDERS = ("baseline0", "baseline1", "ours")


def load_existing_eval(repo_root: Path):
    eval_path = repo_root / "scripts" / "eval" / "droidw_compare_eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)
    spec = importlib.util.spec_from_file_location("existing_droidw_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lower_is_better_improvement(base: float, value: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    return (base - value) / base * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    root_dir = args.root_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root_dir / "eval_three_way"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_eval = load_existing_eval(repo_root)

    results = {}
    rows = []
    for method, folder in zip(METHODS, FOLDERS):
        scene_dir = root_dir / folder
        res = existing_eval.evaluate_trajectory(scene_dir)
        results[method] = res
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
    metrics_df.to_csv(output_dir / "ba_ablation_metrics.csv", index=False)

    base_row = metrics_df.iloc[0]
    improvement_rows = []
    for _, row in metrics_df.iterrows():
        improvement_rows.append(
            {
                "Method": row["Method"],
                "ATE RMSE improvement vs Baseline 0 [%]": lower_is_better_improvement(
                    float(base_row["ATE RMSE [m] ↓"]),
                    float(row["ATE RMSE [m] ↓"]),
                ),
                "RPE Trans RMSE improvement vs Baseline 0 [%]": lower_is_better_improvement(
                    float(base_row["RPE Trans RMSE [m] ↓"]),
                    float(row["RPE Trans RMSE [m] ↓"]),
                ),
                "RPE Rot RMSE improvement vs Baseline 0 [%]": lower_is_better_improvement(
                    float(base_row["RPE Rot RMSE [deg] ↓"]),
                    float(row["RPE Rot RMSE [deg] ↓"]),
                ),
            }
        )
    pd.DataFrame(improvement_rows).to_csv(
        output_dir / "ba_ablation_improvement.csv", index=False
    )

    # XY trajectory plot.
    gt = results["Baseline 0"]["gt_xyz"]
    plt.figure(figsize=(7, 6))
    plt.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=2)
    for method in METHODS:
        xyz = results[method]["est_aligned"]
        n = min(len(gt), len(xyz))
        plt.plot(xyz[:n, 0], xyz[:n, 1], label=method)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("BA ablation: camera trajectory")
    plt.legend()
    plt.savefig(output_dir / "trajectory_xy_three_way.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Per-frame ATE plot.
    plt.figure(figsize=(8, 4))
    for method in METHODS:
        plt.plot(results[method]["ate_errors"], label=method)
    plt.grid(True)
    plt.xlabel("Frame index")
    plt.ylabel("ATE error [m]")
    plt.title("BA ablation: ATE over frames")
    plt.legend()
    plt.savefig(output_dir / "ate_error_three_way.png", dpi=220, bbox_inches="tight")
    plt.close()

    # RPE translation plot.
    plt.figure(figsize=(8, 4))
    for method in METHODS:
        plt.plot(results[method]["rpe_trans_errors"], label=method)
    plt.grid(True)
    plt.xlabel("Frame-pair index")
    plt.ylabel("RPE translation error [m]")
    plt.title("BA ablation: RPE translation")
    plt.legend()
    plt.savefig(output_dir / "rpe_translation_three_way.png", dpi=220, bbox_inches="tight")
    plt.close()

    # RPE rotation plot.
    plt.figure(figsize=(8, 4))
    for method in METHODS:
        plt.plot(results[method]["rpe_rot_errors"], label=method)
    plt.grid(True)
    plt.xlabel("Frame-pair index")
    plt.ylabel("RPE rotation error [deg]")
    plt.title("BA ablation: RPE rotation")
    plt.legend()
    plt.savefig(output_dir / "rpe_rotation_three_way.png", dpi=220, bbox_inches="tight")
    plt.close()

    summary = {
        method: {
            "frames": int(results[method]["frames"]),
            "ate": results[method]["ate"],
            "rpe_trans": results[method]["rpe_trans"],
            "rpe_rot": results[method]["rpe_rot"],
            "scale": float(results[method]["scale"]),
        }
        for method in METHODS
    }
    (output_dir / "ba_ablation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 180)
    print("\n==== Baseline 0 / Baseline 1 / Ours ====")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
