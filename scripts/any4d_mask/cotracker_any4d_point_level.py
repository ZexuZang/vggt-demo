#!/usr/bin/env python3
"""
Pure point-level CoTracker + Any4D dynamic point classifier.

This script does NOT convert dynamic points into a dense smooth mask.
It only outputs point-level tracks and point-level dynamic labels.

Core idea:
    1. CoTracker tracks representative 2D points across frames.
    2. Any4D provides scene-flow / motion magnitude on the reference frame.
    3. We sample Any4D motion at each CoTracker reference point.
    4. A point is dynamic if:
          sampled Any4D motion score is high
          and the point has sufficient CoTracker visibility.
    5. Save point-level labels, scores, tracks, and point-only visualizations.

No GrabCut.
No RGB edge refinement.
No circle rasterization into dense blobs.
No Gaussian blur.
No dynamic prior fusion by default.
"""
import argparse
from pathlib import Path
import csv
import json
import numpy as np
import cv2
import torch


def sorted_images(image_dir: Path, max_frames=None, stride=1):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(image_dir.glob(ext))
    paths = sorted(paths)[::stride]
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")
    return paths


def read_and_resize_frames(image_dir, max_frames=None, stride=1, resize_max_side=768):
    paths = sorted_images(Path(image_dir), max_frames=max_frames, stride=stride)

    bgr_frames, rgb_frames = [], []
    original_hw = None
    resize_scale = 1.0

    for idx, path in enumerate(paths):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        if idx == 0:
            original_hw = img_bgr.shape[:2]
            h, w = original_hw
            if resize_max_side is not None and resize_max_side > 0 and max(h, w) > resize_max_side:
                resize_scale = resize_max_side / float(max(h, w))

        if resize_scale != 1.0:
            new_w = int(round(img_bgr.shape[1] * resize_scale))
            new_h = int(round(img_bgr.shape[0] * resize_scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        bgr_frames.append(img_bgr)
        rgb_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    return paths, bgr_frames, rgb_frames, original_hw, resize_scale


def robust_normalize(x, p_low=50.0, p_high=95.0, eps=1e-6):
    x = np.asarray(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def load_any4d_motion_map(any4d_npz, aggregate="max"):
    data = np.load(any4d_npz, allow_pickle=True)

    if "scene_flow_ref_to_targets" in data:
        flow = data["scene_flow_ref_to_targets"].astype(np.float32)  # M,H,W,3
        mag = np.linalg.norm(flow, axis=-1)  # M,H,W
        if aggregate == "max":
            motion_mag = np.max(mag, axis=0)
        elif aggregate == "mean":
            motion_mag = np.mean(mag, axis=0)
        elif aggregate == "median":
            motion_mag = np.median(mag, axis=0)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")
        return motion_mag, "scene_flow_ref_to_targets"

    if "motion_masks" in data:
        mm = data["motion_masks"].astype(np.float32)
        motion = np.max(mm, axis=0)
        if motion.ndim == 3 and motion.shape[-1] == 1:
            motion = motion[..., 0]
        return motion, "motion_masks"

    if "ref_motion_mask" in data:
        motion = data["ref_motion_mask"].astype(np.float32)
        if motion.ndim == 3 and motion.shape[-1] == 1:
            motion = motion[..., 0]
        return motion, "ref_motion_mask"

    raise KeyError("Any4D npz must contain scene_flow_ref_to_targets, motion_masks, or ref_motion_mask")


def bilinear_sample_map(map2d, xy, src_hw):
    map2d = np.asarray(map2d).astype(np.float32)
    h_m, w_m = map2d.shape[:2]
    h_s, w_s = src_hw

    x = xy[:, 0] * (w_m - 1) / max(w_s - 1, 1)
    y = xy[:, 1] * (h_m - 1) / max(h_s - 1, 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w_m - 1)
    y1 = np.clip(y0 + 1, 0, h_m - 1)
    x0 = np.clip(x0, 0, w_m - 1)
    y0 = np.clip(y0, 0, h_m - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    same = (x0 == x1) & (y0 == y1)
    wa[same] = 1.0
    wb[same] = 0.0
    wc[same] = 0.0
    wd[same] = 0.0

    return (
        wa * map2d[y0, x0]
        + wb * map2d[y1, x0]
        + wc * map2d[y0, x1]
        + wd * map2d[y1, x1]
    ).astype(np.float32)


def run_cotracker(video_rgb, grid_size, device, model_name="cotracker3_offline"):
    frames = np.stack(video_rgb, axis=0)  # T,H,W,3
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)  # 1,T,3,H,W

    print(f"[CoTracker] Loading {model_name} from torch.hub")
    cotracker = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
    cotracker.eval()

    print(f"[CoTracker] Running: video={tuple(video.shape)}, grid_size={grid_size}")
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)

    tracks = pred_tracks[0].detach().float().cpu().numpy()  # T,N,2
    visibility = pred_visibility[0].detach().float().cpu().numpy()
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    return tracks, visibility


def write_point_csv(path, points_xy, points_xy_original, scores, labels, visible_ratio):
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_id",
            "x_resized",
            "y_resized",
            "x_original",
            "y_original",
            "dynamic_score",
            "dynamic_label",
            "visible_ratio",
        ])
        for i, (xy, xy_orig, score, label, vis) in enumerate(
            zip(points_xy, points_xy_original, scores, labels, visible_ratio)
        ):
            writer.writerow([i, float(xy[0]), float(xy[1]), float(xy_orig[0]), float(xy_orig[1]), float(score), int(label), float(vis)])


def save_gray(path, x):
    x = np.asarray(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    cv2.imwrite(str(path), (x * 255).astype(np.uint8))


def draw_points_overlay(img_bgr, points_xy, labels, scores, radius=3):
    out = img_bgr.copy()
    for (x, y), label, score in zip(points_xy, labels, scores):
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= out.shape[1] or y_i < 0 or y_i >= out.shape[0]:
            continue
        color = (0, 0, 255) if label else (0, 255, 0)
        cv2.circle(out, (x_i, y_i), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_dynamic_tracks_overlay(img_bgr, tracks_tn2, labels, max_dynamic_tracks=250):
    out = img_bgr.copy()
    dynamic_ids = np.where(labels > 0)[0]
    if len(dynamic_ids) > max_dynamic_tracks:
        rng = np.random.default_rng(0)
        dynamic_ids = rng.choice(dynamic_ids, max_dynamic_tracks, replace=False)

    for idx in dynamic_ids:
        pts = tracks_tn2[:, idx, :]
        for t in range(1, pts.shape[0]):
            p0 = tuple(np.round(pts[t - 1]).astype(int))
            p1 = tuple(np.round(pts[t]).astype(int))
            if (
                0 <= p0[0] < out.shape[1] and 0 <= p0[1] < out.shape[0]
                and 0 <= p1[0] < out.shape[1] and 0 <= p1[1] < out.shape[0]
            ):
                cv2.line(out, p0, p1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        p_init = tuple(np.round(pts[0]).astype(int))
        if 0 <= p_init[0] < out.shape[1] and 0 <= p_init[1] < out.shape[0]:
            cv2.circle(out, p_init, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return out


def point_level_mask(points_xy, labels, hw, radius=1):
    """
    Point-level mask only. This is not a dense object mask.
    It draws tiny dots at dynamic point locations.
    """
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y), label in zip(points_xy, labels):
        if not label:
            continue
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
            continue
        if radius <= 0:
            mask[y_i, x_i] = 255
        else:
            cv2.circle(mask, (x_i, y_i), radius, 255, -1, lineType=cv2.LINE_AA)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--any4d_npz", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize_max_side", type=int, default=768)
    parser.add_argument("--grid_size", type=int, default=45)
    parser.add_argument("--cotracker_model", default="cotracker3_offline")

    parser.add_argument("--aggregate", choices=["max", "mean", "median"], default="max")
    parser.add_argument("--norm_low_percentile", type=float, default=50.0)
    parser.add_argument("--norm_high_percentile", type=float, default=95.0)

    parser.add_argument("--point_dynamic_thresh", type=float, default=0.55)
    parser.add_argument("--min_visible_ratio", type=float, default=0.35)

    parser.add_argument("--point_vis_radius", type=int, default=3)
    parser.add_argument("--point_mask_radius", type=int, default=1)
    parser.add_argument("--max_dynamic_tracks_draw", type=int, default=250)

    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, bgr_frames, rgb_frames, original_hw, resize_scale = read_and_resize_frames(
        args.image_dir,
        max_frames=args.max_frames,
        stride=args.stride,
        resize_max_side=args.resize_max_side,
    )

    track_h, track_w = rgb_frames[0].shape[:2]
    ref_bgr = bgr_frames[0]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device = torch.device(device)

    tracks, visibility = run_cotracker(
        rgb_frames,
        grid_size=args.grid_size,
        device=device,
        model_name=args.cotracker_model,
    )

    initial_points = tracks[0].astype(np.float32)  # N,2 in resized coordinates
    initial_points_original = initial_points / resize_scale if resize_scale != 0 else initial_points.copy()

    motion_mag, motion_source_key = load_any4d_motion_map(args.any4d_npz, aggregate=args.aggregate)
    motion_score_map = robust_normalize(
        motion_mag,
        p_low=args.norm_low_percentile,
        p_high=args.norm_high_percentile,
    )

    point_motion_score = bilinear_sample_map(
        motion_score_map,
        initial_points,
        src_hw=(track_h, track_w),
    )

    visible_ratio = visibility.mean(axis=0).astype(np.float32)

    point_dynamic_prob = point_motion_score * np.clip(
        visible_ratio / max(args.min_visible_ratio, 1e-6),
        0.0,
        1.0,
    )
    point_dynamic_prob = np.clip(point_dynamic_prob, 0.0, 1.0)

    dynamic_labels = (
        (point_dynamic_prob >= args.point_dynamic_thresh)
        & (visible_ratio >= args.min_visible_ratio)
    ).astype(np.uint8)

    np.savez_compressed(
        out_dir / "point_level_dynamic_tracks.npz",
        image_paths=np.array([str(p) for p in paths]),
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(np.float32),
        initial_points_resized=initial_points.astype(np.float32),
        initial_points_original=initial_points_original.astype(np.float32),
        point_motion_score=point_motion_score.astype(np.float32),
        point_dynamic_prob=point_dynamic_prob.astype(np.float32),
        dynamic_labels=dynamic_labels.astype(np.uint8),
        visible_ratio=visible_ratio.astype(np.float32),
        motion_score_map=motion_score_map.astype(np.float32),
        motion_mag=motion_mag.astype(np.float32),
        track_hw=np.array([track_h, track_w], dtype=np.int32),
        original_hw=np.array(original_hw, dtype=np.int32),
        resize_scale=np.array(resize_scale, dtype=np.float32),
    )

    write_point_csv(
        out_dir / "point_level_dynamic_points.csv",
        initial_points,
        initial_points_original,
        point_dynamic_prob,
        dynamic_labels,
        visible_ratio,
    )

    point_mask = point_level_mask(
        initial_points,
        dynamic_labels,
        hw=(track_h, track_w),
        radius=args.point_mask_radius,
    )
    cv2.imwrite(str(out_dir / "point_level_dynamic_mask.png"), point_mask)

    points_overlay = draw_points_overlay(
        ref_bgr,
        initial_points,
        dynamic_labels,
        point_dynamic_prob,
        radius=args.point_vis_radius,
    )
    cv2.imwrite(str(out_dir / "point_level_points_overlay.png"), points_overlay)

    tracks_overlay = draw_dynamic_tracks_overlay(
        ref_bgr,
        tracks,
        dynamic_labels,
        max_dynamic_tracks=args.max_dynamic_tracks_draw,
    )
    cv2.imwrite(str(out_dir / "point_level_dynamic_tracks_overlay.png"), tracks_overlay)

    motion_score_resized = cv2.resize(
        motion_score_map,
        (track_w, track_h),
        interpolation=cv2.INTER_LINEAR,
    )
    save_gray(out_dir / "any4d_motion_score_map_resized.png", motion_score_resized)

    if resize_scale != 1.0:
        orig_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        point_mask_orig = point_level_mask(
            initial_points_original,
            dynamic_labels,
            hw=orig_bgr.shape[:2],
            radius=max(1, int(round(args.point_mask_radius / max(resize_scale, 1e-6)))),
        )
        cv2.imwrite(str(out_dir / "point_level_dynamic_mask_original_size.png"), point_mask_orig)

        points_overlay_orig = draw_points_overlay(
            orig_bgr,
            initial_points_original,
            dynamic_labels,
            point_dynamic_prob,
            radius=max(2, int(round(args.point_vis_radius / max(resize_scale, 1e-6)))),
        )
        cv2.imwrite(str(out_dir / "point_level_points_overlay_original_size.png"), points_overlay_orig)

    summary = {
        "image_dir": str(args.image_dir),
        "any4d_npz": str(args.any4d_npz),
        "output_dir": str(out_dir),
        "num_frames": len(paths),
        "track_hw": [track_h, track_w],
        "original_hw": list(original_hw),
        "resize_scale": float(resize_scale),
        "grid_size": args.grid_size,
        "num_points": int(initial_points.shape[0]),
        "num_dynamic_points": int(dynamic_labels.sum()),
        "dynamic_point_ratio": float(dynamic_labels.mean()),
        "mean_point_motion_score": float(point_motion_score.mean()),
        "mean_visible_ratio": float(visible_ratio.mean()),
        "motion_source_key": motion_source_key,
        "aggregate": args.aggregate,
        "point_dynamic_thresh": args.point_dynamic_thresh,
        "min_visible_ratio": args.min_visible_ratio,
        "outputs": [
            "point_level_dynamic_tracks.npz",
            "point_level_dynamic_points.csv",
            "point_level_dynamic_mask.png",
            "point_level_points_overlay.png",
            "point_level_dynamic_tracks_overlay.png",
            "any4d_motion_score_map_resized.png",
        ],
    }

    (out_dir / "point_level_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[PointLevelCoTrackerAny4D] Saved to {out_dir}")


if __name__ == "__main__":
    main()
