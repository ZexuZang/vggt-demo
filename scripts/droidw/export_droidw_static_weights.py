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

        dynamic_prob = draw_prob_points(
            points_xy=pts,
            scores=point_scores,
            visibility_t=visibility[t],
            image_hw=(h, w),
            point_radius=args.point_radius,
            visibility_thresh=args.visibility_thresh,
        )

        dynamic_prob = apply_blur(dynamic_prob, args.blur_sigma)
        dynamic_prob = np.clip(dynamic_prob, 0.0, 1.0).astype(np.float32)

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