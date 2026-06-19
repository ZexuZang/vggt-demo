#!/usr/bin/env python3
"""
CoTracker + Any4D dynamic representative point mask.

This module replaces RGB edge-aware refinement with a point-tracking strategy:
1. CoTracker tracks representative 2D points across frames.
2. Any4D scene-flow magnitude is sampled at representative points.
3. Points with strong scene flow are treated as dynamic representative points.
4. Dynamic representative points are rasterized into a dynamic probability map.
5. Static BA weight / mask is updated accordingly.

RGB frames are used only as CoTracker input and for visualization. The dynamic
classification is based on Any4D scene-flow sampled on CoTracker points.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2
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
        raise FileNotFoundError(f"No image files found in {image_dir}")
    return paths


def read_and_resize_frames(image_dir, max_frames=None, stride=1, resize_max_side=768):
    paths = sorted_images(Path(image_dir), max_frames=max_frames, stride=stride)
    bgr_frames, rgb_frames = [], []
    first_orig_hw = None
    scale = 1.0
    for idx, p in enumerate(paths):
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        if idx == 0:
            first_orig_hw = img_bgr.shape[:2]
            h, w = first_orig_hw
            if resize_max_side and max(h, w) > resize_max_side:
                scale = resize_max_side / float(max(h, w))
        if scale != 1.0:
            new_w = int(round(img_bgr.shape[1] * scale))
            new_h = int(round(img_bgr.shape[0] * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        bgr_frames.append(img_bgr)
        rgb_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return paths, bgr_frames, rgb_frames, first_orig_hw, scale


def load_prob(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        x = np.load(p).astype(np.float32)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        x = img.astype(np.float32) / 255.0
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0)


def resize_to(x, hw, interp=cv2.INTER_LINEAR):
    h, w = hw
    if x.shape[:2] == (h, w):
        return x
    return cv2.resize(x, (w, h), interpolation=interp)


def robust_normalize(x, p_low=50, p_high=95, eps=1e-6):
    x = np.asarray(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0)


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
    wa[same], wb[same], wc[same], wd[same] = 1.0, 0.0, 0.0, 0.0
    return (wa * map2d[y0, x0] + wb * map2d[y1, x0] + wc * map2d[y0, x1] + wd * map2d[y1, x1]).astype(np.float32)


def save_gray(path, x):
    x = np.clip(x, 0, 1)
    cv2.imwrite(str(path), (x * 255).astype(np.uint8))


def overlay_prob(img_bgr, prob, color_bgr=(0, 0, 255), alpha=0.45):
    prob = np.clip(prob.astype(np.float32), 0, 1)
    if prob.shape[:2] != img_bgr.shape[:2]:
        prob = resize_to(prob, img_bgr.shape[:2])
    color = np.zeros_like(img_bgr, dtype=np.float32)
    color[:, :] = np.array(color_bgr, dtype=np.float32)
    out = img_bgr.astype(np.float32) * (1 - alpha * prob[..., None]) + color * (alpha * prob[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points_overlay(img_bgr, points_xy, labels, scores, radius=3):
    out = img_bgr.copy()
    for (x, y), label, score in zip(points_xy, labels, scores):
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= out.shape[1] or y_i < 0 or y_i >= out.shape[0]:
            continue
        color = (0, 0, 255) if label else (0, 255, 0)
        r = max(radius, int(radius + 3 * float(score)))
        cv2.circle(out, (x_i, y_i), r, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_track_overlay(img_bgr, tracks_tn2, labels, max_tracks=200):
    out = img_bgr.copy()
    dynamic_indices = np.where(labels > 0)[0]
    if len(dynamic_indices) > max_tracks:
        rng = np.random.default_rng(0)
        dynamic_indices = rng.choice(dynamic_indices, max_tracks, replace=False)
    for n in dynamic_indices:
        pts = tracks_tn2[:, n, :].astype(np.float32)
        for t in range(1, pts.shape[0]):
            p0 = tuple(np.round(pts[t - 1]).astype(int))
            p1 = tuple(np.round(pts[t]).astype(int))
            if 0 <= p0[0] < out.shape[1] and 0 <= p0[1] < out.shape[0] and 0 <= p1[0] < out.shape[1] and 0 <= p1[1] < out.shape[0]:
                cv2.line(out, p0, p1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        p_init = tuple(np.round(pts[0]).astype(int))
        cv2.circle(out, p_init, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return out


def rasterize_dynamic_points(points_xy, scores, labels, hw, point_radius=18, blur_sigma=7):
    h, w = hw
    prob = np.zeros((h, w), dtype=np.float32)
    for (x, y), s, label in zip(points_xy, scores, labels):
        if not label:
            continue
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
            continue
        cv2.circle(prob, (x_i, y_i), int(point_radius), float(s), thickness=-1, lineType=cv2.LINE_AA)
    if blur_sigma > 0:
        k = int(blur_sigma * 4 + 1)
        if k % 2 == 0:
            k += 1
        prob = cv2.GaussianBlur(prob, (k, k), blur_sigma)
    return np.clip(prob, 0.0, 1.0)


def load_any4d_motion_map(any4d_npz, aggregate="max"):
    data = np.load(any4d_npz, allow_pickle=True)
    if "scene_flow_ref_to_targets" in data:
        flow = data["scene_flow_ref_to_targets"].astype(np.float32)
        mag = np.linalg.norm(flow, axis=-1)
        if aggregate == "max":
            return np.max(mag, axis=0), "scene_flow_ref_to_targets"
        if aggregate == "mean":
            return np.mean(mag, axis=0), "scene_flow_ref_to_targets"
        return np.median(mag, axis=0), "scene_flow_ref_to_targets"
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
    raise KeyError("Any4D npz must contain scene_flow_ref_to_targets, motion_masks, or ref_motion_mask.")


def run_cotracker(video_rgb, grid_size, device):
    frames = np.stack(video_rgb, axis=0)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    print("[CoTracker] Loading cotracker3_offline from torch.hub")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()
    print(f"[CoTracker] Running grid_size={grid_size}, video={tuple(video.shape)}")
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)
    tracks = pred_tracks[0].detach().float().cpu().numpy()
    vis = pred_visibility[0].detach().float().cpu().numpy()
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]
    return tracks, vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--any4d_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dynamic_prior_path", default=None)
    parser.add_argument("--static_weight_path", default=None)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize_max_side", type=int, default=768)
    parser.add_argument("--grid_size", type=int, default=35)
    parser.add_argument("--aggregate", choices=["max", "mean", "median"], default="max")
    parser.add_argument("--point_dynamic_thresh", type=float, default=0.55)
    parser.add_argument("--min_visible_ratio", type=float, default=0.35)
    parser.add_argument("--point_radius", type=int, default=18)
    parser.add_argument("--blur_sigma", type=float, default=7.0)
    parser.add_argument("--mask_thresh", type=float, default=0.35)
    parser.add_argument("--combine_with_prior", choices=["none", "max", "mean"], default="max")
    parser.add_argument("--static_weight_thresh", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, bgr_frames, rgb_frames, first_orig_hw, scale = read_and_resize_frames(
        args.image_dir, args.max_frames, args.stride, args.resize_max_side
    )
    track_h, track_w = rgb_frames[0].shape[:2]
    ref_bgr = bgr_frames[0]
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    tracks, visibility = run_cotracker(rgb_frames, args.grid_size, device)
    initial_points = tracks[0]

    motion_mag, motion_key = load_any4d_motion_map(args.any4d_npz, args.aggregate)
    motion_score_map = robust_normalize(motion_mag)
    point_motion_score = bilinear_sample_map(motion_score_map, initial_points, src_hw=(track_h, track_w))

    if args.dynamic_prior_path:
        prior = load_prob(args.dynamic_prior_path)
        prior = resize_to(prior, (track_h, track_w), cv2.INTER_LINEAR)
        point_prior_score = bilinear_sample_map(prior, initial_points, src_hw=(track_h, track_w))
    else:
        prior = None
        point_prior_score = np.zeros_like(point_motion_score)

    visible_ratio = visibility.mean(axis=0).astype(np.float32)
    point_dynamic_score = point_motion_score.copy()
    if prior is not None:
        if args.combine_with_prior == "max":
            point_dynamic_score = np.maximum(point_dynamic_score, point_prior_score)
        elif args.combine_with_prior == "mean":
            point_dynamic_score = 0.5 * point_dynamic_score + 0.5 * point_prior_score

    point_dynamic_prob = point_dynamic_score * np.clip(visible_ratio / max(args.min_visible_ratio, 1e-6), 0.0, 1.0)
    dynamic_labels = ((point_dynamic_prob >= args.point_dynamic_thresh) & (visible_ratio >= args.min_visible_ratio)).astype(np.uint8)

    point_prob = rasterize_dynamic_points(initial_points, point_dynamic_prob, dynamic_labels, (track_h, track_w), args.point_radius, args.blur_sigma)
    if prior is not None:
        if args.combine_with_prior == "max":
            dynamic_prob = np.maximum(point_prob, prior)
        elif args.combine_with_prior == "mean":
            dynamic_prob = 0.5 * point_prob + 0.5 * prior
        else:
            dynamic_prob = point_prob
    else:
        dynamic_prob = point_prob
    dynamic_prob = np.clip(dynamic_prob, 0, 1)
    dynamic_mask = (dynamic_prob >= args.mask_thresh).astype(np.uint8)

    if args.static_weight_path:
        static_weight = np.load(args.static_weight_path).astype(np.float32)
        static_weight = resize_to(static_weight, (track_h, track_w), cv2.INTER_LINEAR)
        static_weight = np.clip(static_weight, 0, 1)
    else:
        static_weight = 1.0 - dynamic_prob
    static_weight_updated = static_weight * (1.0 - dynamic_prob)
    static_ba_mask = (static_weight_updated > args.static_weight_thresh).astype(np.uint8)

    np.save(out_dir / "cotracker_any4d_dynamic_probability.npy", dynamic_prob.astype(np.float32))
    np.save(out_dir / "cotracker_any4d_dynamic_mask.npy", dynamic_mask.astype(np.uint8))
    np.save(out_dir / "cotracker_any4d_static_weight.npy", static_weight_updated.astype(np.float32))
    np.save(out_dir / "cotracker_any4d_static_ba_mask.npy", static_ba_mask.astype(np.uint8))
    np.savez_compressed(
        out_dir / "cotracker_any4d_tracks.npz",
        image_paths=np.array([str(p) for p in paths]),
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(np.float32),
        initial_points=initial_points.astype(np.float32),
        point_motion_score=point_motion_score.astype(np.float32),
        point_prior_score=point_prior_score.astype(np.float32),
        visible_ratio=visible_ratio.astype(np.float32),
        point_dynamic_prob=point_dynamic_prob.astype(np.float32),
        dynamic_labels=dynamic_labels.astype(np.uint8),
        track_hw=np.array([track_h, track_w], dtype=np.int32),
        original_hw=np.array(first_orig_hw, dtype=np.int32),
        resize_scale=np.array(scale, dtype=np.float32),
    )

    save_gray(out_dir / "cotracker_any4d_dynamic_probability.png", dynamic_prob)
    cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_mask.png"), dynamic_mask * 255)
    save_gray(out_dir / "cotracker_any4d_static_weight.png", static_weight_updated)
    cv2.imwrite(str(out_dir / "cotracker_any4d_static_ba_mask.png"), static_ba_mask * 255)
    cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_overlay.png"), overlay_prob(ref_bgr, dynamic_prob, (0, 0, 255), 0.45))
    cv2.imwrite(str(out_dir / "cotracker_any4d_static_ba_overlay.png"), overlay_prob(ref_bgr, static_ba_mask.astype(np.float32), (0, 255, 0), 0.35))
    cv2.imwrite(str(out_dir / "representative_points_overlay.png"), draw_points_overlay(ref_bgr, initial_points, dynamic_labels, point_dynamic_prob))
    cv2.imwrite(str(out_dir / "dynamic_tracks_overlay.png"), draw_track_overlay(ref_bgr, tracks, dynamic_labels))

    if scale != 1.0:
        orig_ref = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        prob_orig = resize_to(dynamic_prob, orig_ref.shape[:2], cv2.INTER_LINEAR)
        mask_orig = resize_to(dynamic_mask.astype(np.float32), orig_ref.shape[:2], cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_overlay_original_size.png"), overlay_prob(orig_ref, prob_orig, (0, 0, 255), 0.45))
        cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_mask_original_size.png"), (mask_orig > 0.5).astype(np.uint8) * 255)

    summary = {
        "image_dir": str(args.image_dir),
        "any4d_npz": str(args.any4d_npz),
        "dynamic_prior_path": str(args.dynamic_prior_path) if args.dynamic_prior_path else None,
        "static_weight_path": str(args.static_weight_path) if args.static_weight_path else None,
        "output_dir": str(out_dir),
        "num_frames": len(paths),
        "track_hw": [track_h, track_w],
        "original_hw": list(first_orig_hw),
        "resize_scale": float(scale),
        "grid_size": args.grid_size,
        "num_representative_points": int(initial_points.shape[0]),
        "num_dynamic_points": int(dynamic_labels.sum()),
        "dynamic_point_ratio": float(dynamic_labels.mean()),
        "dynamic_mask_ratio": float(dynamic_mask.mean()),
        "static_ba_ratio": float(static_ba_mask.mean()),
        "motion_source_key": motion_key,
        "point_dynamic_thresh": args.point_dynamic_thresh,
        "min_visible_ratio": args.min_visible_ratio,
        "point_radius": args.point_radius,
        "blur_sigma": args.blur_sigma,
        "mask_thresh": args.mask_thresh,
        "combine_with_prior": args.combine_with_prior,
    }
    (out_dir / "cotracker_any4d_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[CoTrackerAny4D] Saved to {out_dir}")


if __name__ == "__main__":
    main()
