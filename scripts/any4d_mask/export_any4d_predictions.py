#!/usr/bin/env python3
"""
Export Any4D predictions to an .npz file.

This script calls the official Any4D demo inference functions and saves
scene flow / motion outputs for mask refinement.

Important:
- VGGT repo has a folder named scripts/
- Any4D repo also has a folder named scripts/
- Therefore, we must force Python to import:
      third_party/Any4D/scripts/demo_inference.py
  instead of:
      vggt-demo/scripts/...
"""

import argparse
from pathlib import Path
import sys
import os
import json
import importlib.util

import numpy as np
import torch


def sorted_images(image_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for e in exts:
        paths.extend(image_dir.glob(e))

    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    return paths


def to_numpy(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()

    return np.asarray(x)


def squeeze_batch(x):
    if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == 1:
        return x[0]

    return x


def prepare_any4d_import(any4d_root: Path):
    """
    Make sure that:
        from scripts.demo_inference import ...
    imports Any4D/scripts/demo_inference.py,
    not vggt-demo/scripts.
    """

    any4d_root = any4d_root.resolve()

    if not any4d_root.exists():
        raise FileNotFoundError(f"Any4D root not found: {any4d_root}")

    demo_file = any4d_root / "scripts" / "demo_inference.py"

    if not demo_file.exists():
        raise FileNotFoundError(
            f"Any4D demo_inference.py not found: {demo_file}"
        )

    # Change working directory to Any4D root.
    # This helps Any4D internal relative paths.
    os.chdir(str(any4d_root))

    # Put Any4D root at the very beginning of sys.path.
    any4d_root_str = str(any4d_root)

    while any4d_root_str in sys.path:
        sys.path.remove(any4d_root_str)

    sys.path.insert(0, any4d_root_str)

    # Remove possibly cached wrong scripts package.
    for name in list(sys.modules.keys()):
        if name == "scripts" or name.startswith("scripts."):
            del sys.modules[name]

    print("[Any4D Import] cwd:", Path.cwd())
    print("[Any4D Import] sys.path[0]:", sys.path[0])
    print("[Any4D Import] demo_inference:", demo_file)

def load_any4d_demo_functions(any4d_root: Path):
    """
    Load Any4D/scripts/demo_inference.py by absolute file path.

    This avoids conflict between:
        vggt-demo/scripts
    and:
        Any4D/scripts
    """
    demo_file = any4d_root / "scripts" / "demo_inference.py"

    if not demo_file.exists():
        raise FileNotFoundError(f"demo_inference.py not found: {demo_file}")

    module_name = "any4d_demo_inference_absolute"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(demo_file))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {demo_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "init_inference_model"):
        raise AttributeError("demo_inference.py has no init_inference_model")

    if not hasattr(module, "sample_inference"):
        raise AttributeError("demo_inference.py has no sample_inference")

    print("[Any4D Import] Loaded demo_inference by absolute path:")
    print("[Any4D Import]", demo_file)

    return module.init_inference_model, module.sample_inference

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--any4d_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_npz", required=True)

    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--machine", default="local")

    parser.add_argument("--ref_img_idx", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=8)

    parser.add_argument("--resolution_w", type=int, default=518)
    parser.add_argument("--resolution_h", type=int, default=336)

    parser.add_argument("--ref_img_binary_mask_path", default=None)
    parser.add_argument("--use_scene_flow_mask_refined", action="store_true")

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    any4d_root = Path(args.any4d_root).resolve()
    image_dir = Path(args.image_dir).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_npz = Path(args.output_npz).resolve()

    config_dir = (
        Path(args.config_dir).resolve()
        if args.config_dir is not None
        else any4d_root / "configs"
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    prepare_any4d_import(any4d_root)

    init_inference_model, sample_inference = load_any4d_demo_functions(any4d_root)

    from any4d.utils.image import load_images
    from any4d.utils.moge_inference import load_moge_model

    image_paths_all = sorted_images(image_dir)

    if args.ref_img_idx < 0 or args.ref_img_idx >= len(image_paths_all):
        raise IndexError(
            f"ref_img_idx={args.ref_img_idx} outside image list length {len(image_paths_all)}"
        )

    start = max(0, args.start_idx)
    end = min(len(image_paths_all), args.end_idx)

    if end <= start:
        raise ValueError(
            f"end_idx must be greater than start_idx, got start={start}, end={end}"
        )

    ref_path = image_paths_all[args.ref_img_idx]

    target_indices = [
        i for i in range(start, end)
        if i != args.ref_img_idx
    ]

    target_paths = [
        image_paths_all[i]
        for i in target_indices
    ]

    image_list = [str(ref_path)] + [str(p) for p in target_paths]

    device_name = (
        args.device
        if torch.cuda.is_available() and args.device == "cuda"
        else "cpu"
    )

    device = torch.device(device_name)

    high_level_config = {
        "path": str(config_dir / "train.yaml"),
        "config_overrides": [
            f"machine={args.machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": str(checkpoint_path),
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }

    print("[Any4D] any4d_root:", any4d_root)
    print("[Any4D] image_dir:", image_dir)
    print("[Any4D] num_all_images:", len(image_paths_all))
    print("[Any4D] selected_images:", len(image_list))
    print("[Any4D] ref_img_idx:", args.ref_img_idx)
    print("[Any4D] target_indices:", target_indices)
    print("[Any4D] checkpoint:", checkpoint_path)
    print("[Any4D] output_npz:", output_npz)
    print("[Any4D] device:", device)

    print("[Any4D] Loading model")
    model = init_inference_model(
        high_level_config,
        high_level_config["checkpoint_path"],
        device,
    )

    print("[Any4D] Loading MoGe helper model")
    moge_model = load_moge_model(device=str(device))

    print(f"[Any4D] Loading {len(image_list)} images")
    views = load_images(
        image_list,
        size=(args.resolution_w, args.resolution_h),
        verbose=False,
        norm_type=high_level_config["data_norm_type"],
        patch_size=14,
        compute_moge_mask=True,
        moge_model=moge_model,
        binary_mask_path=args.ref_img_binary_mask_path,
    )

    print("[Any4D] Inference")
    pred_result = sample_inference(
        model,
        views,
        device,
        use_amp=high_level_config["trained_with_amp"],
    )

    key_report = {}

    for i in range(len(views)):
        pred_key = f"pred{i + 1}"

        if pred_key in pred_result:
            key_report[pred_key] = sorted(list(pred_result[pred_key].keys()))

    print("[Any4D] Prediction keys")
    print(json.dumps(key_report, indent=2))

    arrays = {
        "image_paths": np.array(image_list),
        "ref_img_idx": np.array(args.ref_img_idx, dtype=np.int32),
        "target_original_indices": np.array(target_indices, dtype=np.int32),
    }

    pred1 = pred_result.get("pred1", {})

    for key in [
        "pts3d",
        "pts3d_cam",
        "ray_directions",
        "depth_along_ray",
        "motion_mask",
    ]:
        if key in pred1:
            arrays[f"ref_{key}"] = to_numpy(squeeze_batch(pred1[key]))

    scene_flows = []
    flow_target_local_indices = []
    motion_masks = []

    for i in range(1, len(views)):
        pred = pred_result.get(f"pred{i + 1}", {})

        if "scene_flow" in pred:
            scene_flows.append(
                to_numpy(squeeze_batch(pred["scene_flow"]))
            )
            flow_target_local_indices.append(i)

        if "motion_mask" in pred:
            motion_masks.append(
                to_numpy(squeeze_batch(pred["motion_mask"]))
            )

    if scene_flows:
        arrays["scene_flow_ref_to_targets"] = np.stack(scene_flows, axis=0)
        arrays["flow_target_local_indices"] = np.array(
            flow_target_local_indices,
            dtype=np.int32,
        )

    if motion_masks:
        arrays["motion_masks"] = np.stack(motion_masks, axis=0)

    # Save additional useful small tensors for debugging.
    for pred_name, pred in pred_result.items():
        if not isinstance(pred, dict):
            continue

        for key, val in pred.items():
            if key in [
                "scene_flow",
                "motion_mask",
                "pts3d",
                "pts3d_cam",
            ]:
                continue

            if torch.is_tensor(val) and val.numel() < 5_000_000:
                arrays[f"{pred_name}_{key}"] = to_numpy(squeeze_batch(val))

    np.savez_compressed(output_npz, **arrays)

    keys_json = output_npz.with_suffix(".keys.json")
    meta_json = output_npz.with_suffix(".json")

    keys_json.write_text(json.dumps(key_report, indent=2))

    shape_report = {
        k: list(v.shape)
        for k, v in arrays.items()
        if hasattr(v, "shape")
    }

    meta_json.write_text(
        json.dumps(
            {
                "output_npz": str(output_npz),
                "image_list": image_list,
                "keys": list(arrays.keys()),
                "shapes": shape_report,
                "prediction_keys": key_report,
            },
            indent=2,
        )
    )

    print(f"[Any4D] Saved {output_npz}")
    print("[Any4D] Shape report:")
    print(json.dumps(shape_report, indent=2))


if __name__ == "__main__":
    main()