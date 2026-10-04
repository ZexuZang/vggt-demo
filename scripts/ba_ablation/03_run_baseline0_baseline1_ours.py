#!/usr/bin/env python3
"""
Run the same DROID-W scene with three BA variants without overwriting results.

Baseline 0
    Original DROID-W config; ours_dynamic disabled.

Baseline 1
    Current external-weight mechanism with scalar refined-mask weight files.

Ours
    Same mechanism after 02_patch_droidw_for_anisotropic_weight.py, using
    2-channel [x,y] precision files.

Example
-------
python scripts/ba_ablation/03_run_baseline0_baseline1_ours.py \
  --repo_root /content/vggt-demo \
  --weights_root /content/vggt-demo/outputs/ba_ablation_weights/person_tracking \
  --output_root /content/vggt-demo/outputs_compare/ba_ablation
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


VARIANTS = ("baseline0", "baseline1", "ours")


def write_external_config(
    config_path: Path,
    base_config_from_droid_root: str,
    weight_dir: Path,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""inherit_from: ./{base_config_from_droid_root}

ours_dynamic:
  enable: true
  static_weight_dir: {weight_dir}
  sequential_names: true
  alpha: 1.0
  min_weight: 0.0
"""
    )


def build_runtime_env(droid_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(droid_root) + ":" + env.get("PYTHONPATH", "")

    # Same idea as the existing Colab notebook: prepend torch/lib for CUDA-linked extensions.
    try:
        torch_lib = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
            ],
            text=True,
        ).strip()
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    except Exception as exc:
        print(f"[WARN] Could not resolve torch/lib: {exc}")

    return env


def run_variant(
    droid_root: Path,
    config_path: Path,
    native_scene_output: Path,
    saved_output: Path,
) -> None:
    if native_scene_output.exists():
        shutil.rmtree(native_scene_output)

    env = build_runtime_env(droid_root)
    cmd = [sys.executable, "run.py", "--config", str(config_path)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=droid_root, env=env, check=True)

    if not native_scene_output.exists():
        raise FileNotFoundError(
            f"DROID-W finished but native output was not found: {native_scene_output}"
        )

    if saved_output.exists():
        shutil.rmtree(saved_output)
    saved_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(native_scene_output, saved_output)
    print(f"[SAVE] {saved_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument(
        "--base_config",
        default="configs/Dynamic/Bonn/bonn_person_tracking.yaml",
        help="Path relative to third_party/DROID-W.",
    )
    parser.add_argument("--weights_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument(
        "--native_scene_output",
        default="Outputs/Bonn/bonn_person_tracking",
        help="DROID-W native output path relative to third_party/DROID-W.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    droid_root = repo_root / "third_party" / "DROID-W"
    if not (droid_root / "run.py").exists():
        raise FileNotFoundError(droid_root / "run.py")

    weights_root = args.weights_root.resolve()
    baseline1_weight_dir = weights_root / "baseline1" / "static_weight_npy"
    ours_weight_dir = weights_root / "ours" / "precision_weight_npy"

    if "baseline1" in args.variants and not baseline1_weight_dir.exists():
        raise FileNotFoundError(
            f"Baseline 1 weights not found: {baseline1_weight_dir}. "
            "Run 01_generate_ba_ablation_weights.py first."
        )
    if "ours" in args.variants and not ours_weight_dir.exists():
        raise FileNotFoundError(
            f"Ours weights not found: {ours_weight_dir}. "
            "Run 01_generate_ba_ablation_weights.py first."
        )

    generated_cfg_dir = droid_root / "configs" / "generated_ba_ablation"
    baseline1_cfg = generated_cfg_dir / "baseline1_soft_isotropic.yaml"
    ours_cfg = generated_cfg_dir / "ours_motion_mahalanobis.yaml"

    write_external_config(baseline1_cfg, args.base_config, baseline1_weight_dir)
    write_external_config(ours_cfg, args.base_config, ours_weight_dir)

    configs = {
        "baseline0": droid_root / args.base_config,
        "baseline1": baseline1_cfg,
        "ours": ours_cfg,
    }

    native_scene_output = droid_root / args.native_scene_output
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print("\n" + "=" * 76)
        print(f"BA VARIANT: {variant}")
        print("=" * 76)
        run_variant(
            droid_root=droid_root,
            config_path=configs[variant],
            native_scene_output=native_scene_output,
            saved_output=output_root / variant,
        )

    print("\n[DONE]")
    for variant in args.variants:
        print(f"{variant:10s}: {output_root / variant}")


if __name__ == "__main__":
    main()
