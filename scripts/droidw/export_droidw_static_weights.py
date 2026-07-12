#!/usr/bin/env python3
"""
Export per-frame static weights for DROID-W from point-level dynamic tracks.

Input:
    cotracker_any4d_tracks.npz

Expected useful keys:
    tracks:              [T, N, 2]
    visibility:          [T, N]
    dynamic_labels:      [N]
    point_dynamic_prob:  [N]
    track_hw:            [2], optional
    original_hw:         [2], optional

Output:
    output_dir/
        dynamic_prob_npy/
            000000.npy
            000001.npy
            ...
        static_weight_npy/
            000000.npy
            000001.npy
            ...
        dynamic_prob_png/
            000000.png
            ...
        static_weight_png/
            000000.png
            ...
        overlay_dynamic/
            000000.png
            ...
        summary.json

Meaning:
    dynamic_prob = probability that pixel/point belongs to dynamic object
    static_weight = 1 - dynamic_prob

For DROID-W:
    dynamic region should have lower BA weight.
    static_weight can be multiplied with DROID-W original confidence/weight.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sorted_images(image_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(image_dir.glob(ext))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_track_npz(path: Path):
    data = np.load(path, allow_pickle=True)

    required = ["tracks", "visibility", "dynamic_labels"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}. Available keys: {list(data.keys())}")

    tracks = data["tracks"].astype(np.float32)          # T,N,2
    visibility = data["visibility"].astype(np.float32)  # T,N
    dynamic_labels = data["dynamic_labels"].astype(np.uint8)  # N

    if "point_dynamic_prob" in data:
        point_dynamic_prob = data["point_dynamic_prob"].astype(np.float32)
    elif "point_motion_score" in data:
        point_dynamic_prob = data["point_motion_score"].astype(np.float32)
    else:
        print("[WARN] No point_dynamic_prob or point_motion_score found. Use dynamic_labels as score.")
        point_dynamic_prob = dynamic_labels.astype(np.float32)

    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks should be [T,N,2], got {tracks.shape}")

    if visibility.shape[:2] != tracks.shape[:2]:
        raise ValueError(f"visibility shape {visibility.shape} does not match tracks {tracks.shape}")

    if dynamic_labels.shape[0] != tracks.shape[1]:
        raise ValueError(
            f"dynamic_labels length {dynamic_labels.shape[0]} does not match num points {tracks.shape[1]}"
        )

    if point_dynamic_prob.shape[0] != tracks.shape[1]:
        raise ValueError(
            f"point_dynamic_prob length {point_dynamic_prob.shape[0]} does not match num points {tracks.shape[1]}"
        )

    track_hw = None
    if "track_hw" in data:
        track_hw = tuple(int(x) for x in data["track_hw"].tolist())

    original_hw = None
    if "original_hw" in data:
        original_hw = tuple(int(x) for x in data["original_hw"].tolist())

    return {
        "tracks": tracks,
        "visibility": visibility,
        "dynamic_labels": dynamic_labels,
        "point_dynamic_prob": np.clip(point_dynamic_prob, 0.0, 1.0),
        "track_hw": track_hw,
        "original_hw": original_hw,
        "keys": list(data.keys()),
    }


def draw_prob_points(
    points_xy,
    scores,
    visibility_t,
    image_hw,
    point_radius=3,
    visibility_thresh=0.5,
):
    """
    Rasterize dynamic point scores into a per-frame dynamic probability map.

    points_xy: [N,2] in current image coordinate
    scores: [N], dynamic probability for each point
    visibility_t: [N], CoTracker visibility for current frame
    image_hw: (H,W)
    """
    h, w = image_hw
    prob = np.zeros((h, w), dtype=np.float32)

    for (x, y), s, v in zip(points_xy, scores, visibility_t):
        if v < visibility_thresh:
            continue

        if s <= 0:
            continue

        xi = int(round(float(x)))
        yi = int(round(float(y)))

        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        s = float(np.clip(s, 0.0, 1.0))

        if point_radius <= 0:
            prob[yi, xi] = max(prob[yi, xi], s)
        else:
            # Draw into a temporary mask to use max blending.
            tmp = np.zeros_like(prob, dtype=np.float32)
            cv2.circle(tmp, (xi, yi), int(point_radius), s, -1, lineType=cv2.LINE_AA)
            prob = np.maximum(prob, tmp)

    return prob


def apply_blur(prob, blur_sigma):
    if blur_sigma is None or blur_sigma <= 0:
        return prob

    # Kernel size should be odd and roughly 6*sigma.
    k = int(round(blur_sigma * 6 + 1))
    if k % 2 == 0:
        k += 1
    k = max(k, 3)

    blurred = cv2.GaussianBlur(prob.astype(np.float32), (k, k), blur_sigma)
    return np.clip(blurred, 0.0, 1.0)

def sorted_refined_prob_files(refined_path: Path):
    """
    Accept either:
    1. a single refined dynamic probability file, e.g.
       cotracker_any4d_dynamic_probability.npy
    2. a directory containing per-frame refined dynamic probability maps.
    """
    if refined_path is None:
        return []

    refined_path = Path(refined_path)

    if refined_path.is_file():
        return [refined_path]

    if not refined_path.exists():
        raise FileNotFoundError(refined_path)

    files = []
    for ext in ["*.npy", "*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        files.extend(refined_path.glob(ext))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No refined dynamic probability files found in {refined_path}")

    return files


def load_prob_map(path: Path):
    """
    Load a probability map in [0, 1].
    Prefer .npy. PNG is supported only for grayscale masks/probability maps.
    Do not use overlay images here.
    """
    path = Path(path)

    if path.suffix.lower() == ".npy":
        x = np.load(path).astype(np.float32)

        if x.ndim == 3:
            if x.shape[-1] == 1:
                x = x[..., 0]
            elif x.shape[0] == 1:
                x = x[0]
            else:
                # fallback for H,W,C probability-like data
                x = x.mean(axis=-1).astype(np.float32)

    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x = img.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0

    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def preprocess_refined_dynamic(
    dyn,
    image_hw,
    gamma=1.0,
    dynamic_scale=1.0,
    binary_thresh=-1.0,
    dilate_radius=0,
    blur_sigma=0.0,
):
    """
    Convert refined dynamic probability to a DROID-W dynamic probability map.
    gamma < 1 strengthens weak dynamic probability.
    dynamic_scale > 1 increases dynamic suppression.
    dilation / blur makes BA weight smoother and more object-level.
    """
    h, w = image_hw

    if dyn.shape[:2] != (h, w):
        dyn = cv2.resize(dyn.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    dyn = np.clip(dyn.astype(np.float32), 0.0, 1.0)

    if gamma != 1.0:
        dyn = np.power(dyn, gamma).astype(np.float32)

    dyn = np.clip(dyn * dynamic_scale, 0.0, 1.0)

    if binary_thresh is not None and binary_thresh >= 0:
        dyn = (dyn >= binary_thresh).astype(np.float32)

    if dilate_radius is not None and dilate_radius > 0:
        k = int(dilate_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dyn = cv2.dilate(dyn, kernel, iterations=1)

    if blur_sigma is not None and blur_sigma > 0:
        dyn = apply_blur(dyn, blur_sigma)

    return np.clip(dyn, 0.0, 1.0).astype(np.float32)


def save_gray_png(path: Path, x):
    x = np.clip(x, 0.0, 1.0)
    cv2.imwrite(str(path), (x * 255).astype(np.uint8))


def overlay_dynamic(img_bgr, dynamic_prob, alpha=0.45):
    if dynamic_prob.shape[:2] != img_bgr.shape[:2]:
        dynamic_prob = cv2.resize(
            dynamic_prob,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    m = np.clip(dynamic_prob.astype(np.float32), 0.0, 1.0)
    red = np.zeros_like(img_bgr, dtype=np.float32)
    red[:, :] = np.array([0, 0, 255], dtype=np.float32)

    out = img_bgr.astype(np.float32) * (1.0 - alpha * m[..., None]) + red * (alpha * m[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tracks_npz", required=True, help="Path to cotracker_any4d_tracks.npz")
    parser.add_argument("--image_dir", required=True, help="Image folder used by DROID-W / original sequence")
    parser.add_argument("--output_dir", required=True, help="Output folder for per-frame static weights")

    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--visibility_thresh", type=float, default=0.5)

    parser.add_argument("--point_radius", type=int, default=3)
    parser.add_argument("--blur_sigma", type=float, default=1.5)

    parser.add_argument("--min_static_weight", type=float, default=0.05)
    parser.add_argument("--dynamic_scale", type=float, default=1.0)
    parser.add_argument(
        "--refined_dynamic_path",
        default=None,
        help="Path to refined dense dynamic probability map or directory. "
            "If provided, it can replace or combine with point-rasterized dynamic probability.",
    )

    parser.add_argument(
        "--refined_mode",
        choices=["replace", "max", "mean"],
        default="replace",
        help="How to use refined dynamic probability. "
            "replace: use refined map directly; "
            "max: max(point rasterization, refined map); "
            "mean: average them.",
    )

    parser.add_argument(
        "--repeat_last_refined",
        action="store_true",
        help="If refined_dynamic_path has fewer maps than frames, repeat the last refined map.",
    )

    parser.add_argument(
        "--refined_gamma",
        type=float,
        default=1.0,
        help="Gamma for refined dynamic probability. gamma < 1 strengthens weak dynamic regions.",
    )

    parser.add_argument(
        "--refined_dynamic_scale",
        type=float,
        default=1.0,
        help="Scale refined dynamic probability before converting to static weight.",
    )

    parser.add_argument(
        "--refined_binary_thresh",
        type=float,
        default=-1.0,
        help="If >=0, binarize refined dynamic probability before dilation/blur.",
    )

    parser.add_argument(
        "--refined_dilate_radius",
        type=int,
        default=0,
        help="Dilate refined dynamic region before computing static weight.",
    )

    parser.add_argument(
        "--refined_blur_sigma",
        type=float,
        default=0.0,
        help="Smooth refined dynamic probability before computing static weight.",
    )

    parser.add_argument(
        "--use_sequential_names",
        action="store_true",
        help="Save as 000000.npy, 000001.npy. Otherwise use original image stem.",
    )

    args = parser.parse_args()

    tracks_npz = Path(args.tracks_npz)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)

    out_dynamic_npy = out_dir / "dynamic_prob_npy"
    out_static_npy = out_dir / "static_weight_npy"
    out_dynamic_png = out_dir / "dynamic_prob_png"
    out_static_png = out_dir / "static_weight_png"
    out_overlay = out_dir / "overlay_dynamic"

    for d in [out_dynamic_npy, out_static_npy, out_dynamic_png, out_static_png, out_overlay]:
        ensure_dir(d)

    pack = load_track_npz(tracks_npz)

    tracks = pack["tracks"]
    visibility = pack["visibility"]
    dynamic_labels = pack["dynamic_labels"]
    point_dynamic_prob = pack["point_dynamic_prob"]

    image_paths = sorted_images(image_dir)

    refined_files = []
    if args.refined_dynamic_path is not None:
        refined_files = sorted_refined_prob_files(Path(args.refined_dynamic_path))
        print(f"[INFO] refined_dynamic_path: {args.refined_dynamic_path}")
        print(f"[INFO] num_refined_dynamic_maps={len(refined_files)}")

    T_tracks = tracks.shape[0]
    T_images = len(image_paths)
    T = min(T_tracks, T_images)

    if args.max_frames is not None:
        T = min(T, args.max_frames)

    print(f"[INFO] tracks_npz: {tracks_npz}")
    print(f"[INFO] image_dir:  {image_dir}")
    print(f"[INFO] output_dir: {out_dir}")
    print(f"[INFO] T_tracks={T_tracks}, T_images={T_images}, exporting T={T}")
    print(f"[INFO] num_points={tracks.shape[1]}")
    print(f"[INFO] dynamic_points={int(dynamic_labels.sum())}")
    print(f"[INFO] track_hw={pack['track_hw']}, original_hw={pack['original_hw']}")
    print(f"[INFO] keys={pack['keys']}")
    print(f"[INFO] refined_mode={args.refined_mode}")
    
    # Only dynamic points contribute to dynamic probability.
    point_scores = point_dynamic_prob * dynamic_labels.astype(np.float32)
    point_scores = np.clip(point_scores * args.dynamic_scale, 0.0, 1.0)

    exported = []

    for t in range(T):
        img_path = image_paths[t]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        h, w = img_bgr.shape[:2]

        pts = tracks[t].copy().astype(np.float32)

        # If CoTracker was run on resized frames, map points back to original image size.
        if pack["track_hw"] is not None:
            track_h, track_w = pack["track_hw"]
            if (track_h, track_w) != (h, w):
                sx = w / max(float(track_w), 1.0)
                sy = h / max(float(track_h), 1.0)
                pts[:, 0] *= sx
                pts[:, 1] *= sy
        else:
            # Fallback: infer from coordinate range, usually not needed.
            max_x = np.nanmax(pts[:, 0])
            max_y = np.nanmax(pts[:, 1])
            if max_x <= 0.8 * w or max_y <= 0.8 * h:
                print(
                    f"[WARN] Frame {t}: track_hw not found and point range seems smaller than image. "
                    "If output is misaligned, save track_hw in the source npz."
                )

        # 1. Original point-level rasterization.
        point_dynamic_prob_map = draw_prob_points(
            points_xy=pts,
            scores=point_scores,
            visibility_t=visibility[t],
            image_hw=(h, w),
            point_radius=args.point_radius,
            visibility_thresh=args.visibility_thresh,
        )

        point_dynamic_prob_map = apply_blur(point_dynamic_prob_map, args.blur_sigma)
        point_dynamic_prob_map = np.clip(point_dynamic_prob_map, 0.0, 1.0).astype(np.float32)

        # 2. Optional refined dense dynamic probability.
        if args.refined_dynamic_path is not None:
            if len(refined_files) == 1:
                refined_file = refined_files[0]
            elif t < len(refined_files):
                refined_file = refined_files[t]
            elif args.repeat_last_refined:
                refined_file = refined_files[-1]
            else:
                raise RuntimeError(
                    f"Not enough refined dynamic maps: frame {t}, "
                    f"num_refined={len(refined_files)}. "
                    f"Use --repeat_last_refined if this is intended."
                )

            refined_dynamic_prob = load_prob_map(refined_file)
            refined_dynamic_prob = preprocess_refined_dynamic(
                refined_dynamic_prob,
                image_hw=(h, w),
                gamma=args.refined_gamma,
                dynamic_scale=args.refined_dynamic_scale,
                binary_thresh=args.refined_binary_thresh,
                dilate_radius=args.refined_dilate_radius,
                blur_sigma=args.refined_blur_sigma,
            )

            if args.refined_mode == "replace":
                dynamic_prob = refined_dynamic_prob
            elif args.refined_mode == "max":
                dynamic_prob = np.maximum(point_dynamic_prob_map, refined_dynamic_prob)
            elif args.refined_mode == "mean":
                dynamic_prob = 0.5 * point_dynamic_prob_map + 0.5 * refined_dynamic_prob
            else:
                raise ValueError(f"Unknown refined_mode: {args.refined_mode}")

        else:
            refined_file = None
            refined_dynamic_prob = None
            dynamic_prob = point_dynamic_prob_map

        dynamic_prob = np.clip(dynamic_prob, 0.0, 1.0).astype(np.float32)

        # 3. Convert dynamic probability to static BA weight.
        static_weight = 1.0 - dynamic_prob
        static_weight = np.clip(static_weight, args.min_static_weight, 1.0).astype(np.float32)

        if args.use_sequential_names:
            stem = f"{t:06d}"
        else:
            stem = img_path.stem

        np.save(out_dynamic_npy / f"{stem}.npy", dynamic_prob)
        np.save(out_static_npy / f"{stem}.npy", static_weight)

        save_gray_png(out_dynamic_png / f"{stem}.png", dynamic_prob)
        save_gray_png(out_static_png / f"{stem}.png", static_weight)

        overlay = overlay_dynamic(img_bgr, dynamic_prob, alpha=0.45)
        cv2.imwrite(str(out_overlay / f"{stem}.png"), overlay)

        exported.append(
            {
                "frame_index": t,
                "image": str(img_path),
                "stem": stem,
                "dynamic_ratio_soft": float(dynamic_prob.mean()),
                "dynamic_ratio_hard_05": float((dynamic_prob > 0.5).mean()),
                "static_weight_mean": float(static_weight.mean()),
                "static_weight_min": float(static_weight.min()),
                "refined_file": str(refined_file) if refined_file is not None else None,
                "point_dynamic_mean": float(point_dynamic_prob_map.mean()),
                "final_dynamic_mean": float(dynamic_prob.mean()),
                "final_dynamic_ratio_hard_05": float((dynamic_prob > 0.5).mean()),
            }
        )

        if t % 10 == 0 or t == T - 1:
            print(
                f"[{t:04d}/{T}] {img_path.name}: "
                f"dyn_mean={dynamic_prob.mean():.4f}, "
                f"static_mean={static_weight.mean():.4f}"
            )

    summary = {
        "tracks_npz": str(tracks_npz),
        "image_dir": str(image_dir),
        "output_dir": str(out_dir),
        "num_exported_frames": T,
        "num_points": int(tracks.shape[1]),
        "num_dynamic_points": int(dynamic_labels.sum()),
        "point_radius": args.point_radius,
        "blur_sigma": args.blur_sigma,
        "visibility_thresh": args.visibility_thresh,
        "min_static_weight": args.min_static_weight,
        "dynamic_scale": args.dynamic_scale,
        "track_hw": pack["track_hw"],
        "refined_dynamic_path": str(args.refined_dynamic_path) if args.refined_dynamic_path else None,
        "refined_mode": args.refined_mode,
        "repeat_last_refined": bool(args.repeat_last_refined),
        "refined_gamma": args.refined_gamma,
        "refined_dynamic_scale": args.refined_dynamic_scale,
        "refined_binary_thresh": args.refined_binary_thresh,
        "refined_dilate_radius": args.refined_dilate_radius,
        "refined_blur_sigma": args.refined_blur_sigma,
        "num_refined_dynamic_maps": len(refined_files),
        "original_hw": pack["original_hw"],
        "outputs": {
            "dynamic_prob_npy": str(out_dynamic_npy),
            "static_weight_npy": str(out_static_npy),
            "dynamic_prob_png": str(out_dynamic_png),
            "static_weight_png": str(out_static_png),
            "overlay_dynamic": str(out_overlay),
        },
        "frames": exported,
    }

    ensure_dir(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Saved per-frame DROID-W weights to: {out_dir}")


if __name__ == "__main__":
    main()