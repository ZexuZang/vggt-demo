#!/usr/bin/env python3
"""
Fuse Any4D scene-flow motion cues with VGGT confidence.

Outputs:
- dynamic_probability.npy / .png
- static_weight.npy / .png
- refined_dynamic_mask.png
- static_ba_mask.png
- exclude_from_static_ba_mask.png
- overlays

This is the core experiment:
Any4D provides motion-aware dynamic probability.
VGGT confidence preserves high-confidence static regions.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def resize_map(x, hw, interpolation=cv2.INTER_LINEAR):
    h, w = hw
    x = np.asarray(x)
    if x.shape[:2] == (h, w):
        return x
    return cv2.resize(x, (w, h), interpolation=interpolation)


def normalize_confidence(conf, eps=1e-6):
    conf = np.asarray(conf).astype(np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    if conf.ndim == 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    p10, p90 = np.percentile(conf, [10, 90])
    return np.clip((conf - p10) / (p90 - p10 + eps), 0.0, 1.0)


def robust_dynamic_probability(motion_mag, sharpness=8.0, eps=1e-6):
    mag = np.asarray(motion_mag).astype(np.float32)
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    p50, p95 = np.percentile(mag, [50, 95])
    scaled = np.clip((mag - p50) / (p95 - p50 + eps), 0.0, 1.0)

    # Convert robust normalized motion into probability.
    return sigmoid(sharpness * (scaled - 0.5)).astype(np.float32)


def morph_mask(mask, kernel_size=5, close_iter=1, open_iter=1):
    if kernel_size <= 1:
        return mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = mask.astype(np.uint8) * 255
    if close_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iter)
    if open_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iter)
    return (out > 0).astype(np.uint8)


def load_reference_image(image_dir, ref_idx):
    image_dir = Path(image_dir)
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for e in exts:
        paths.extend(image_dir.glob(e))
    paths = sorted(paths)
    if not paths:
        return None, None
    idx = min(max(ref_idx, 0), len(paths) - 1)
    img = cv2.imread(str(paths[idx]), cv2.IMREAD_COLOR)
    if img is None:
        return None, paths[idx]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), paths[idx]


def save_heatmap(path, x):
    x = np.asarray(x).astype(np.float32)
    x = np.nan_to_num(x)
    x = np.clip(x, 0, 1)
    cv2.imwrite(str(path), (x * 255).astype(np.uint8))


def overlay_mask(rgb, mask_or_prob, color=(255, 0, 0), alpha=0.45):
    rgb = rgb.copy().astype(np.float32)
    m = np.asarray(mask_or_prob).astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0, 1)
    if m.shape[:2] != rgb.shape[:2]:
        m = resize_map(m, rgb.shape[:2])
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out = rgb * (1 - alpha * m[..., None]) + color_arr * (alpha * m[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt_npz", required=True)
    parser.add_argument("--any4d_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--ref_idx", type=int, default=0)
    parser.add_argument("--aggregate", choices=["max", "mean", "median"], default="max")
    parser.add_argument("--dynamic_thresh", type=float, default=0.55)
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--static_weight_thresh", type=float, default=0.45)
    parser.add_argument("--morph_kernel", type=int, default=5)
    parser.add_argument("--dynamic_boost_from_uncertainty", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vggt = np.load(args.vggt_npz, allow_pickle=True)
    any4d = np.load(args.any4d_npz, allow_pickle=True)

    if "depth" in vggt:
        ref_depth = vggt["depth"][args.ref_idx]
        if ref_depth.ndim == 3 and ref_depth.shape[-1] == 1:
            ref_depth = ref_depth[..., 0]
        target_hw = ref_depth.shape[:2]
    else:
        raise KeyError("VGGT npz must contain depth")

    if "scene_flow_ref_to_targets" in any4d:
        flow = any4d["scene_flow_ref_to_targets"].astype(np.float32)  # M,H,W,3
        mag = np.linalg.norm(flow, axis=-1)
        if args.aggregate == "max":
            motion_mag = np.max(mag, axis=0)
        elif args.aggregate == "mean":
            motion_mag = np.mean(mag, axis=0)
        else:
            motion_mag = np.median(mag, axis=0)
    elif "motion_masks" in any4d:
        mm = any4d["motion_masks"].astype(np.float32)
        motion_mag = np.max(mm, axis=0)
        if motion_mag.ndim == 3 and motion_mag.shape[-1] == 1:
            motion_mag = motion_mag[..., 0]
    elif "ref_motion_mask" in any4d:
        motion_mag = any4d["ref_motion_mask"].astype(np.float32)
        if motion_mag.ndim == 3 and motion_mag.shape[-1] == 1:
            motion_mag = motion_mag[..., 0]
    else:
        raise KeyError(
            "Any4D npz must contain scene_flow_ref_to_targets, motion_masks, or ref_motion_mask"
        )

    motion_mag = resize_map(motion_mag, target_hw, interpolation=cv2.INTER_LINEAR)
    motion_prob = robust_dynamic_probability(motion_mag)

    conf_maps = []
    if "depth_conf" in vggt:
        conf_maps.append(normalize_confidence(vggt["depth_conf"][args.ref_idx]))
    if "point_conf" in vggt:
        conf_maps.append(normalize_confidence(vggt["point_conf"][args.ref_idx]))

    if conf_maps:
        conf_maps = [resize_map(c, target_hw, interpolation=cv2.INTER_LINEAR) for c in conf_maps]
        vggt_conf = np.sqrt(np.prod(np.stack(conf_maps, axis=0), axis=0)).astype(np.float32)
    else:
        vggt_conf = np.ones(target_hw, dtype=np.float32)

    uncertainty = 1.0 - vggt_conf

    dynamic_prob = np.clip(
        motion_prob + args.dynamic_boost_from_uncertainty * uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)

    static_weight = ((1.0 - motion_prob) * vggt_conf).astype(np.float32)

    refined_dynamic_mask = dynamic_prob > args.dynamic_thresh
    refined_dynamic_mask = morph_mask(refined_dynamic_mask, args.morph_kernel)

    static_ba_mask = static_weight > args.static_weight_thresh
    static_ba_mask = morph_mask(static_ba_mask, args.morph_kernel, close_iter=1, open_iter=0)

    exclude_mask = (dynamic_prob > args.dynamic_thresh) | (vggt_conf < args.conf_thresh)
    exclude_mask = morph_mask(exclude_mask, args.morph_kernel)

    np.save(out_dir / "motion_magnitude.npy", motion_mag)
    np.save(out_dir / "motion_probability.npy", motion_prob)
    np.save(out_dir / "vggt_confidence.npy", vggt_conf)
    np.save(out_dir / "dynamic_probability.npy", dynamic_prob)
    np.save(out_dir / "static_weight.npy", static_weight)

    save_heatmap(out_dir / "motion_probability.png", motion_prob)
    save_heatmap(out_dir / "vggt_confidence.png", vggt_conf)
    save_heatmap(out_dir / "dynamic_probability.png", dynamic_prob)
    save_heatmap(out_dir / "static_weight.png", static_weight)

    cv2.imwrite(str(out_dir / "refined_dynamic_mask.png"), refined_dynamic_mask * 255)
    cv2.imwrite(str(out_dir / "static_ba_mask.png"), static_ba_mask * 255)
    cv2.imwrite(str(out_dir / "exclude_from_static_ba_mask.png"), exclude_mask * 255)

    image_path = None
    if args.image_dir:
        rgb, image_path = load_reference_image(args.image_dir, args.ref_idx)
        if rgb is not None:
            rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
            dyn_overlay = overlay_mask(rgb, dynamic_prob, color=(255, 0, 0), alpha=0.45)
            static_overlay = overlay_mask(rgb, static_ba_mask, color=(0, 255, 0), alpha=0.35)
            cv2.imwrite(str(out_dir / "refined_dynamic_overlay.png"), cv2.cvtColor(dyn_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "static_ba_overlay.png"), cv2.cvtColor(static_overlay, cv2.COLOR_RGB2BGR))

    summary = {
        "target_hw": list(target_hw),
        "ref_idx": args.ref_idx,
        "image_path": str(image_path) if image_path else None,
        "aggregate": args.aggregate,
        "dynamic_thresh": args.dynamic_thresh,
        "conf_thresh": args.conf_thresh,
        "static_weight_thresh": args.static_weight_thresh,
        "dynamic_ratio": float(refined_dynamic_mask.mean()),
        "static_ba_ratio": float(static_ba_mask.mean()),
        "exclude_ratio": float(exclude_mask.mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[MaskRefine] Saved results to {out_dir}")


if __name__ == "__main__":
    main()
