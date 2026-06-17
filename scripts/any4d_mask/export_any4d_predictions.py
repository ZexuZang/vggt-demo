#!/usr/bin/env python3
"""
Export Any4D predictions to an .npz file.

This script calls the official Any4D demo inference functions and saves
scene flow / motion outputs for mask refinement.

Run inside the Any4D environment.

If Any4D changes output key names, this script also writes:
    any4d_pred.keys.json
so you can inspect the available prediction keys.
"""
import argparse
from pathlib import Path
import sys
import json
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
    sys.path.insert(0, str(any4d_root))
    sys.path.insert(0, str(any4d_root / "scripts"))

    config_dir = Path(args.config_dir) if args.config_dir else any4d_root / "configs"
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from scripts.demo_inference import init_inference_model, sample_inference
    from any4d.utils.image import load_images
    from any4d.utils.moge_inference import load_moge_model

    image_paths_all = sorted_images(Path(args.image_dir))

    if args.ref_img_idx < 0 or args.ref_img_idx >= len(image_paths_all):
        raise IndexError("ref_img_idx outside image list")

    start = max(0, args.start_idx)
    end = min(len(image_paths_all), args.end_idx)
    if end <= start:
        raise ValueError("end_idx must be greater than start_idx")

    ref_path = image_paths_all[args.ref_img_idx]
    target_indices = [i for i in range(start, end) if i != args.ref_img_idx]
    target_paths = [image_paths_all[i] for i in target_indices]
    image_list = [str(ref_path)] + [str(p) for p in target_paths]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device = torch.device(device)

    high_level_config = {
        "path": str(config_dir / "train.yaml"),
        "config_overrides": [
            f"machine={args.machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": args.checkpoint_path,
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }

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
        pred_key = f"pred{i+1}"
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
    for key in ["pts3d", "pts3d_cam", "ray_directions", "depth_along_ray", "motion_mask"]:
        if key in pred1:
            arrays[f"ref_{key}"] = to_numpy(squeeze_batch(pred1[key]))

    scene_flows = []
    flow_target_local_indices = []
    motion_masks = []

    for i in range(1, len(views)):
        pred = pred_result.get(f"pred{i+1}", {})

        if "scene_flow" in pred:
            scene_flows.append(to_numpy(squeeze_batch(pred["scene_flow"])))
            flow_target_local_indices.append(i)

        if "motion_mask" in pred:
            motion_masks.append(to_numpy(squeeze_batch(pred["motion_mask"])))

    if scene_flows:
        arrays["scene_flow_ref_to_targets"] = np.stack(scene_flows, axis=0)
        arrays["flow_target_local_indices"] = np.array(flow_target_local_indices, dtype=np.int32)

    if motion_masks:
        arrays["motion_masks"] = np.stack(motion_masks, axis=0)

    # Fallback: save all small/known useful arrays with readable keys.
    for pred_name, pred in pred_result.items():
        if not isinstance(pred, dict):
            continue
        for key, val in pred.items():
            if key in ["scene_flow", "motion_mask", "pts3d", "pts3d_cam"]:
                continue
            if torch.is_tensor(val) and val.numel() < 5000000:
                arrays[f"{pred_name}_{key}"] = to_numpy(squeeze_batch(val))

    np.savez_compressed(out_path, **arrays)

    out_path.with_suffix(".keys.json").write_text(json.dumps(key_report, indent=2))

    shape_report = {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")}
    out_path.with_suffix(".json").write_text(json.dumps({
        "output_npz": str(out_path),
        "image_list": image_list,
        "keys": list(arrays.keys()),
        "shapes": shape_report,
        "prediction_keys": key_report,
    }, indent=2))

    print(f"[Any4D] Saved {out_path}")
    print(json.dumps(shape_report, indent=2))


if __name__ == "__main__":
    main()
