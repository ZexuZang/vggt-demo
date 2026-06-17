#!/usr/bin/env python3
"""
Export VGGT predictions to an .npz file.

Run inside your VGGT environment.

Expected output keys:
- image_paths
- extrinsic
- intrinsic
- depth
- depth_conf
- point_map
- point_conf
- point_map_by_unprojection
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch


def sorted_images(image_dir: Path, max_frames=None, stride=1):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for e in exts:
        paths.extend(image_dir.glob(e))
    paths = sorted(paths)[::stride]
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="facebook/VGGT-1B")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted_images(image_dir, args.max_frames, args.stride)

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device = torch.device(device)

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"[VGGT] Loading {args.model_name}")
    model = VGGT.from_pretrained(args.model_name).to(device).eval()

    print(f"[VGGT] Loading {len(image_paths)} frames")
    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    images_b = images[None]

    print("[VGGT] Inference")
    with torch.no_grad():
        autocast_enabled = device.type == "cuda"
        with torch.cuda.amp.autocast(dtype=dtype, enabled=autocast_enabled):
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)

            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_b.shape[-2:]
            )

            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_b, ps_idx
            )

            point_map, point_conf = model.point_head(
                aggregated_tokens_list, images_b, ps_idx
            )

        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0),
            extrinsic.squeeze(0),
            intrinsic.squeeze(0),
        )

    arrays = {
        "image_paths": np.array([str(p) for p in image_paths]),
        "preprocessed_hw": np.array(images_b.shape[-2:], dtype=np.int32),
        "extrinsic": to_numpy(extrinsic.squeeze(0)),
        "intrinsic": to_numpy(intrinsic.squeeze(0)),
        "depth": to_numpy(depth_map.squeeze(0)),
        "depth_conf": to_numpy(depth_conf.squeeze(0)) if depth_conf is not None else None,
        "point_map": to_numpy(point_map.squeeze(0)) if point_map is not None else None,
        "point_conf": to_numpy(point_conf.squeeze(0)) if point_conf is not None else None,
        "point_map_by_unprojection": to_numpy(point_map_by_unprojection),
    }
    arrays = {k: v for k, v in arrays.items() if v is not None}

    np.savez_compressed(out_path, **arrays)

    meta = {
        "num_frames": len(image_paths),
        "image_dir": str(image_dir),
        "output_npz": str(out_path),
        "keys": list(arrays.keys()),
        "shapes": {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")},
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[VGGT] Saved {out_path}")
    print(json.dumps(meta["shapes"], indent=2))


if __name__ == "__main__":
    main()
