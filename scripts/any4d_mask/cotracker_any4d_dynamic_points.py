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

def apply_affine(M, pts):
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    return (pts_h @ M.T).astype(np.float32)


def apply_homography(H, pts):
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    pred_h = pts_h @ H.T
    z = pred_h[:, 2:3]
    pred = pred_h[:, :2] / np.maximum(z, 1e-6)
    return pred.astype(np.float32)


def estimate_background_motion_residual(
    tracks,
    visibility,
    point_motion_score,
    bg_model="affine",
    ransac_thresh=3.0,
    static_candidate_percentile=45.0,
    min_bg_points=80,
    vis_thresh=0.5,
):
    """
    Estimate how well each CoTracker point follows the static background motion.

    tracks: T, N, 2
    visibility: T, N
    point_motion_score: N, Any4D scene-flow score sampled at reference points

    Return:
        bg_residual_px: N
        bg_residual_score: N, normalized to [0, 1]
        per_frame_residual: T, N
        bg_model_success: T
    """
    tracks = tracks.astype(np.float32)
    visibility = visibility.astype(np.float32)

    T, N, _ = tracks.shape
    ref_pts = tracks[0]
    visible = visibility > vis_thresh

    low_motion_cut = np.percentile(point_motion_score, static_candidate_percentile)
    low_motion = point_motion_score <= low_motion_cut

    per_frame_residual = np.full((T, N), np.nan, dtype=np.float32)
    bg_model_success = np.zeros((T,), dtype=np.uint8)

    per_frame_residual[0] = 0.0
    bg_model_success[0] = 1

    for t in range(1, T):
        # Prefer low Any4D-motion points as static background candidates.
        cand = low_motion & visible[0] & visible[t]

        # Fallback if too few candidates.
        if int(cand.sum()) < min_bg_points:
            cand = visible[0] & visible[t]

        if int(cand.sum()) < 8:
            print(f"[BGResidual] Frame {t}: too few points, skipped. candidates={int(cand.sum())}")
            continue

        src = ref_pts[cand].astype(np.float32)
        dst = tracks[t, cand].astype(np.float32)

        try:
            if bg_model == "affine":
                M, inliers = cv2.estimateAffinePartial2D(
                    src,
                    dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thresh,
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=10,
                )
                if M is None:
                    print(f"[BGResidual] Frame {t}: affine estimation failed.")
                    continue

                pred = apply_affine(M.astype(np.float32), ref_pts)

            elif bg_model == "homography":
                H, inliers = cv2.findHomography(
                    src,
                    dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thresh,
                    maxIters=2000,
                    confidence=0.99,
                )
                if H is None:
                    print(f"[BGResidual] Frame {t}: homography estimation failed.")
                    continue

                pred = apply_homography(H.astype(np.float32), ref_pts)

            else:
                raise ValueError(f"Unknown bg_model: {bg_model}")

        except cv2.error as e:
            print(f"[BGResidual] Frame {t}: OpenCV error: {e}")
            continue

        res = np.linalg.norm(pred - tracks[t], axis=1).astype(np.float32)
        res[~(visible[0] & visible[t])] = np.nan

        per_frame_residual[t] = res
        bg_model_success[t] = 1

        if inliers is not None:
            inlier_ratio = float(np.asarray(inliers).reshape(-1).mean())
        else:
            inlier_ratio = -1.0

        print(
            f"[BGResidual] Frame {t}: model={bg_model}, "
            f"candidates={int(cand.sum())}, "
            f"inlier_ratio={inlier_ratio:.3f}, "
            f"median_res={np.nanmedian(res):.3f}px"
        )

    with np.errstate(all="ignore"):
        bg_residual_px = np.nanmedian(per_frame_residual[1:], axis=0).astype(np.float32)

    bg_residual_px = np.nan_to_num(bg_residual_px, nan=0.0, posinf=0.0, neginf=0.0)
    bg_residual_score = robust_normalize(bg_residual_px, p_low=50.0, p_high=95.0)

    return bg_residual_px, bg_residual_score, per_frame_residual, bg_model_success


def draw_score_overlay(img_bgr, points_xy, scores, radius=3):
    """
    Visualize per-point scalar score.
    Low score = blue/green, high score = red.
    """
    out = img_bgr.copy()
    scores = np.clip(scores.astype(np.float32), 0.0, 1.0)

    for (x, y), s in zip(points_xy, scores):
        x_i = int(round(x))
        y_i = int(round(y))

        if x_i < 0 or x_i >= out.shape[1] or y_i < 0 or y_i >= out.shape[0]:
            continue

        color_val = np.array([[int(round(s * 255))]], dtype=np.uint8)
        color = cv2.applyColorMap(color_val, cv2.COLORMAP_JET)[0, 0].tolist()

        cv2.circle(out, (x_i, y_i), radius, color, -1, lineType=cv2.LINE_AA)

    return out

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

def load_vggt_depth(vggt_npz, ref_idx=0):
    if vggt_npz is None:
        return None

    data = np.load(vggt_npz, allow_pickle=True)

    if "depth" not in data:
        raise KeyError("VGGT npz must contain key: depth")

    depth = data["depth"][ref_idx].astype(np.float32)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def boxes_from_dynamic_points(points_xy, labels, hw, pad=40, link_radius=24, min_seed_points=3):
    """
    Build local resampling boxes from coarse dynamic points.

    This mask is only used to make query boxes, not as final dynamic mask.
    """
    h, w = hw
    marker = np.zeros((h, w), dtype=np.uint8)

    dyn_pts = points_xy[labels > 0]

    for x, y in dyn_pts:
        x_i = int(round(x))
        y_i = int(round(y))
        if 0 <= x_i < w and 0 <= y_i < h:
            cv2.circle(marker, (x_i, y_i), link_radius, 255, -1)

    num, cc, stats, _ = cv2.connectedComponentsWithStats(marker, connectivity=8)

    boxes = []
    for i in range(1, num):
        ys, xs = np.where(cc == i)
        if len(xs) == 0:
            continue

        # Count how many dynamic seed points fall in this component.
        x0_raw, y0_raw = xs.min(), ys.min()
        x1_raw, y1_raw = xs.max(), ys.max()

        in_comp = (
            (dyn_pts[:, 0] >= x0_raw)
            & (dyn_pts[:, 0] <= x1_raw)
            & (dyn_pts[:, 1] >= y0_raw)
            & (dyn_pts[:, 1] <= y1_raw)
        )

        if int(in_comp.sum()) < min_seed_points:
            continue

        x0 = max(0, int(x0_raw - pad))
        y0 = max(0, int(y0_raw - pad))
        x1 = min(w - 1, int(x1_raw + pad))
        y1 = min(h - 1, int(y1_raw + pad))

        boxes.append((x0, y0, x1, y1))

    return boxes


def make_local_query_points_from_boxes(
    boxes,
    step,
    hw,
    motion_score_map=None,
    motion_gate=0.0,
    depth_map=None,
    seed_points=None,
    seed_labels=None,
    depth_gate=0.25,
):
    """
    Generate local query points inside dynamic boxes.

    Optional filters:
    - motion_score_map gate
    - VGGT depth consistency gate
    """
    h, w = hw
    all_queries = []

    for box in boxes:
        x0, y0, x1, y1 = box

        xs = np.arange(x0, x1 + 1, step, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, step, dtype=np.float32)

        if len(xs) == 0 or len(ys) == 0:
            continue

        grid_x, grid_y = np.meshgrid(xs, ys)
        queries = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)

        # Motion gate: avoid sampling large static areas inside the box.
        if motion_score_map is not None and motion_gate > 0:
            motion_score_resized = resize_to(motion_score_map, hw, cv2.INTER_LINEAR)
            q_motion = bilinear_sample_map(motion_score_resized, queries, src_hw=hw)
            queries = queries[q_motion >= motion_gate]

        # Depth gate: keep points close to dynamic seed depth.
        if depth_map is not None and seed_points is not None and seed_labels is not None:
            dyn_seed = seed_points[seed_labels > 0]

            # Use seed points inside current box.
            in_box = (
                (dyn_seed[:, 0] >= x0)
                & (dyn_seed[:, 0] <= x1)
                & (dyn_seed[:, 1] >= y0)
                & (dyn_seed[:, 1] <= y1)
            )
            dyn_seed_box = dyn_seed[in_box]

            if len(dyn_seed_box) > 0 and len(queries) > 0:
                depth_resized = resize_to(depth_map, hw, cv2.INTER_LINEAR)

                seed_depth = bilinear_sample_map(depth_resized, dyn_seed_box, src_hw=hw)
                seed_depth = seed_depth[seed_depth > 0]

                if len(seed_depth) > 0:
                    median_depth = np.median(seed_depth)

                    q_depth = bilinear_sample_map(depth_resized, queries, src_hw=hw)
                    rel_err = np.abs(q_depth - median_depth) / max(float(median_depth), 1e-6)

                    queries = queries[(q_depth > 0) & (rel_err <= depth_gate)]

        if len(queries) > 0:
            all_queries.append(queries)

    if not all_queries:
        return np.zeros((0, 2), dtype=np.float32)

    queries = np.concatenate(all_queries, axis=0)

    # Remove duplicate integer positions.
    q_int = np.round(queries).astype(np.int32)
    _, unique_idx = np.unique(q_int, axis=0, return_index=True)
    queries = queries[np.sort(unique_idx)]

    return queries.astype(np.float32)


def run_cotracker_with_queries(video_rgb, query_xy, device):
    """
    Run CoTracker using explicit query points.

    query_xy: N,2 in frame-0 pixel coordinates.
    CoTracker query format is usually [t, x, y], with t=0.
    """
    if len(query_xy) == 0:
        return None, None

    frames = np.stack(video_rgb, axis=0)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

    print("[CoTracker] Loading cotracker3_offline for local queries")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()

    t0 = np.zeros((query_xy.shape[0], 1), dtype=np.float32)
    queries = np.concatenate([t0, query_xy.astype(np.float32)], axis=1)

    queries_torch = torch.from_numpy(queries)[None].float().to(device)

    print(f"[CoTracker] Running local queries: {queries_torch.shape[1]} points")

    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, queries=queries_torch)

    tracks = pred_tracks[0].detach().float().cpu().numpy()
    vis = pred_visibility[0].detach().float().cpu().numpy()

    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]

    return tracks, vis

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def depth_aware_connected_component_filter(
    points_xy,
    point_depth,
    point_scores,
    dynamic_labels,
    hw,
    radius=18.0,
    depth_gate=0.25,
    score_gate=0.35,
    min_component_points=3,
    min_dynamic_points=2,
    min_dynamic_ratio=0.25,
    min_mean_score=0.45,
    propagate=False,
):
    """
    Use VGGT depth to build point connected components.

    Two points are connected if:
        1. their 2D distance is small;
        2. their VGGT depth is similar;
        3. their dynamic scores are similar.

    If propagate=False:
        Keep only original dynamic points that belong to valid depth-aware components.
        This is good for removing false positive red points.

    If propagate=True:
        Mark all points inside valid depth-aware components as dynamic.
        This is good for recovering feet / wheels / thin structures.
    """
    points_xy = points_xy.astype(np.float32)
    point_depth = point_depth.astype(np.float32)
    point_scores = point_scores.astype(np.float32)
    dynamic_labels = dynamic_labels.astype(np.uint8)

    h, w = hw
    n = points_xy.shape[0]

    if n == 0:
        return dynamic_labels, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.uint8)

    uf = UnionFind(n)

    # Spatial hash grid for efficient neighbor search.
    cell_size = max(1, int(round(radius)))
    buckets = {}

    for i, (x, y) in enumerate(points_xy):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        cx = int(x // cell_size)
        cy = int(y // cell_size)
        buckets.setdefault((cx, cy), []).append(i)

    valid_depth = np.isfinite(point_depth) & (point_depth > 0)

    for (cx, cy), ids in buckets.items():
        neighbor_ids = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_ids.extend(buckets.get((cx + dx, cy + dy), []))

        for i in ids:
            xi, yi = points_xy[i]
            di = point_depth[i]

            if not valid_depth[i]:
                continue

            for j in neighbor_ids:
                if j <= i:
                    continue

                if not valid_depth[j]:
                    continue

                xj, yj = points_xy[j]
                dj = point_depth[j]

                dist_2d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                if dist_2d > radius:
                    continue

                depth_denom = max((float(di) + float(dj)) * 0.5, 1e-6)
                rel_depth_diff = abs(float(di) - float(dj)) / depth_denom

                if rel_depth_diff > depth_gate:
                    continue

                score_diff = abs(float(point_scores[i]) - float(point_scores[j]))

                if score_gate >= 0 and score_diff > score_gate:
                    continue

                uf.union(i, j)

    # Collect components.
    root_to_ids = {}
    for i in range(n):
        r = uf.find(i)
        root_to_ids.setdefault(r, []).append(i)

    refined_labels = np.zeros_like(dynamic_labels)
    component_ids = np.full((n,), -1, dtype=np.int32)
    component_valid = np.zeros((n,), dtype=np.uint8)

    comp_idx = 0

    for _, ids in root_to_ids.items():
        ids = np.asarray(ids, dtype=np.int32)

        for idx in ids:
            component_ids[idx] = comp_idx

        comp_idx += 1

        if len(ids) < min_component_points:
            continue

        dyn_count = int(dynamic_labels[ids].sum())
        dyn_ratio = dyn_count / max(len(ids), 1)
        mean_score = float(point_scores[ids].mean())

        is_valid_dynamic_component = (
            (dyn_count >= min_dynamic_points)
            and (
                dyn_ratio >= min_dynamic_ratio
                or mean_score >= min_mean_score
            )
        )

        if not is_valid_dynamic_component:
            continue

        component_valid[ids] = 1

        if propagate:
            # Recover weak dynamic points inside the same depth-consistent component.
            refined_labels[ids] = 1
        else:
            # Conservative mode: only keep original dynamic points in valid components.
            refined_labels[ids] = dynamic_labels[ids]

    return refined_labels.astype(np.uint8), component_ids, component_valid

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
    # Local dynamic-region re-sampling.
    parser.add_argument("--local_resample", action="store_true")
    parser.add_argument("--local_step", type=int, default=6)
    parser.add_argument("--local_pad", type=int, default=40)
    parser.add_argument("--local_link_radius", type=int, default=24)
    parser.add_argument("--local_min_seed_points", type=int, default=3)
    parser.add_argument("--local_max_queries", type=int, default=2500)
    parser.add_argument("--local_seed_thresh", type=float, default=0.55)
    parser.add_argument("--local_motion_gate", type=float, default=0.20)

    # Optional VGGT depth gate for local query points.
    parser.add_argument("--vggt_npz", default=None)
    parser.add_argument("--vggt_ref_idx", type=int, default=0)
    parser.add_argument("--local_depth_gate", type=float, default=0.25)
    parser.add_argument("--point_dynamic_thresh", type=float, default=0.55)
    parser.add_argument("--min_visible_ratio", type=float, default=0.35)
    # Depth-aware point connected component.
    parser.add_argument("--use_depth_cc", action="store_true")
    parser.add_argument("--cc_radius", type=float, default=18.0)
    parser.add_argument("--cc_depth_gate", type=float, default=0.25)
    parser.add_argument("--cc_score_gate", type=float, default=0.35)
    parser.add_argument("--cc_min_component_points", type=int, default=3)
    parser.add_argument("--cc_min_dynamic_points", type=int, default=2)
    parser.add_argument("--cc_min_dynamic_ratio", type=float, default=0.25)
    parser.add_argument("--cc_min_mean_score", type=float, default=0.45)
    parser.add_argument("--cc_propagate", action="store_true")
    # Background motion residual filtering.
    parser.add_argument("--use_bg_motion_residual", action="store_true")
    parser.add_argument("--bg_model", choices=["affine", "homography"], default="affine")
    parser.add_argument("--bg_ransac_thresh", type=float, default=3.0)
    parser.add_argument("--bg_static_candidate_percentile", type=float, default=45.0)
    parser.add_argument("--bg_min_points", type=int, default=80)
    parser.add_argument("--bg_vis_thresh", type=float, default=0.5)
    parser.add_argument("--bg_residual_score_thresh", type=float, default=0.45)
    parser.add_argument("--bg_residual_px_thresh", type=float, default=2.0)
    parser.add_argument("--point_radius", type=int, default=18)
    parser.add_argument("--point_vis_radius", type=int, default=3)
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

    # ------------------------------------------------------------
    # Optional local dynamic-region re-sampling
    # ------------------------------------------------------------
    if args.local_resample:
        print("[LocalResample] Building preliminary dynamic seed points")

        prelim_score = point_motion_score.copy()
        prelim_visible_ratio = visibility.mean(axis=0).astype(np.float32)

        prelim_labels = (
            (prelim_score >= args.local_seed_thresh)
            & (prelim_visible_ratio >= args.min_visible_ratio)
        ).astype(np.uint8)

        boxes = boxes_from_dynamic_points(
            initial_points,
            prelim_labels,
            hw=(track_h, track_w),
            pad=args.local_pad,
            link_radius=args.local_link_radius,
            min_seed_points=args.local_min_seed_points,
        )

        print(f"[LocalResample] Found {len(boxes)} local dynamic boxes: {boxes}")

        vggt_depth = load_vggt_depth(args.vggt_npz, args.vggt_ref_idx) if args.vggt_npz else None

        local_queries = make_local_query_points_from_boxes(
            boxes=boxes,
            step=args.local_step,
            hw=(track_h, track_w),
            motion_score_map=motion_score_map,
            motion_gate=args.local_motion_gate,
            depth_map=vggt_depth,
            seed_points=initial_points,
            seed_labels=prelim_labels,
            depth_gate=args.local_depth_gate,
        )

        if len(local_queries) > args.local_max_queries:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(local_queries), args.local_max_queries, replace=False)
            local_queries = local_queries[keep]

        print(f"[LocalResample] Local query points after filters: {len(local_queries)}")

        if len(local_queries) > 0:
            local_tracks, local_visibility = run_cotracker_with_queries(
                rgb_frames,
                local_queries,
                device,
            )

            if local_tracks is not None:
                tracks = np.concatenate([tracks, local_tracks], axis=1)
                visibility = np.concatenate([visibility, local_visibility], axis=1)
                initial_points = tracks[0]

                # Recompute Any4D score for all global + local points.
                point_motion_score = bilinear_sample_map(
                    motion_score_map,
                    initial_points,
                    src_hw=(track_h, track_w),
                )

                print(
                    f"[LocalResample] Total points after local resampling: "
                    f"{initial_points.shape[0]}"
                )

    if args.dynamic_prior_path:
        prior = load_prob(args.dynamic_prior_path)
        prior = resize_to(prior, (track_h, track_w), cv2.INTER_LINEAR)
        point_prior_score = bilinear_sample_map(prior, initial_points, src_hw=(track_h, track_w))
    else:
        prior = None
        point_prior_score = np.zeros_like(point_motion_score)

    visible_ratio = visibility.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------
    # 1. Combine Any4D scene-flow score and optional prior score
    # ------------------------------------------------------------
    point_dynamic_score = point_motion_score.copy()

    if prior is not None:
        if args.combine_with_prior == "max":
            point_dynamic_score = np.maximum(point_dynamic_score, point_prior_score)
        elif args.combine_with_prior == "mean":
            point_dynamic_score = 0.5 * point_dynamic_score + 0.5 * point_prior_score


    # ------------------------------------------------------------
    # 2. Optional background motion residual
    #    This filters points that have high Any4D score but still
    #    follow the estimated static background motion.
    # ------------------------------------------------------------
    if args.use_bg_motion_residual:
        bg_residual_px, bg_residual_score, per_frame_bg_residual, bg_model_success = (
            estimate_background_motion_residual(
                tracks=tracks,
                visibility=visibility,
                point_motion_score=point_dynamic_score,
                bg_model=args.bg_model,
                ransac_thresh=args.bg_ransac_thresh,
                static_candidate_percentile=args.bg_static_candidate_percentile,
                min_bg_points=args.bg_min_points,
                vis_thresh=args.bg_vis_thresh,
            )
        )
    else:
        bg_residual_px = np.zeros_like(point_dynamic_score, dtype=np.float32)
        bg_residual_score = np.ones_like(point_dynamic_score, dtype=np.float32)
        per_frame_bg_residual = np.full(
            (tracks.shape[0], tracks.shape[1]),
            np.nan,
            dtype=np.float32,
        )
        bg_model_success = np.zeros((tracks.shape[0],), dtype=np.uint8)


    # ------------------------------------------------------------
    # 3. Point-level dynamic decision
    # ------------------------------------------------------------
    visibility_weight = np.clip(
        visible_ratio / max(args.min_visible_ratio, 1e-6),
        0.0,
        1.0,
    )

    if args.use_bg_motion_residual:
        point_dynamic_prob = point_dynamic_score * visibility_weight * bg_residual_score
    else:
        point_dynamic_prob = point_dynamic_score * visibility_weight

    point_dynamic_prob = np.clip(point_dynamic_prob, 0.0, 1.0).astype(np.float32)

    scene_flow_condition = point_dynamic_score >= args.point_dynamic_thresh
    visibility_condition = visible_ratio >= args.min_visible_ratio

    if args.use_bg_motion_residual:
        bg_condition = (
            (bg_residual_score >= args.bg_residual_score_thresh)
            & (bg_residual_px >= args.bg_residual_px_thresh)
        )
    else:
        bg_condition = np.ones_like(scene_flow_condition, dtype=bool)

    dynamic_labels = (
        scene_flow_condition
        & visibility_condition
        & bg_condition
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # 3.5 Optional depth-aware point connected component
    # ------------------------------------------------------------
    dynamic_labels_before_cc = dynamic_labels.copy()

    if args.use_depth_cc:
        if args.vggt_npz is None:
            raise ValueError("--use_depth_cc requires --vggt_npz")

        vggt_depth = load_vggt_depth(args.vggt_npz, args.vggt_ref_idx)
        vggt_depth_resized = resize_to(vggt_depth, (track_h, track_w), cv2.INTER_LINEAR)

        point_depth = bilinear_sample_map(
            vggt_depth_resized,
            initial_points,
            src_hw=(track_h, track_w),
        )

        dynamic_labels, depth_cc_ids, depth_cc_valid = depth_aware_connected_component_filter(
            points_xy=initial_points,
            point_depth=point_depth,
            point_scores=point_dynamic_prob,
            dynamic_labels=dynamic_labels,
            hw=(track_h, track_w),
            radius=args.cc_radius,
            depth_gate=args.cc_depth_gate,
            score_gate=args.cc_score_gate,
            min_component_points=args.cc_min_component_points,
            min_dynamic_points=args.cc_min_dynamic_points,
            min_dynamic_ratio=args.cc_min_dynamic_ratio,
            min_mean_score=args.cc_min_mean_score,
            propagate=args.cc_propagate,
        )

        print(
            "[DepthCC] dynamic points before/after: "
            f"{int(dynamic_labels_before_cc.sum())} -> {int(dynamic_labels.sum())}"
        )

    else:
        point_depth = np.zeros_like(point_dynamic_prob, dtype=np.float32)
        depth_cc_ids = np.full((initial_points.shape[0],), -1, dtype=np.int32)
        depth_cc_valid = np.zeros((initial_points.shape[0],), dtype=np.uint8)

    # ------------------------------------------------------------
    # 4. Optional dense visualization/output
    #    This part still makes a dense mask. If you want pure point-level
    #    output, delete this block or set point_radius=0 and blur_sigma=0.
    # ------------------------------------------------------------
    point_prob = rasterize_dynamic_points(
        initial_points,
        point_dynamic_prob,
        dynamic_labels,
        (track_h, track_w),
        args.point_radius,
        args.blur_sigma,
    )

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
        point_dynamic_score=point_dynamic_score.astype(np.float32),
        visible_ratio=visible_ratio.astype(np.float32),
        point_dynamic_prob=point_dynamic_prob.astype(np.float32),
        dynamic_labels=dynamic_labels.astype(np.uint8),
        dynamic_labels_before_cc=dynamic_labels_before_cc.astype(np.uint8),
        point_depth=point_depth.astype(np.float32),
        depth_cc_ids=depth_cc_ids.astype(np.int32),
        depth_cc_valid=depth_cc_valid.astype(np.uint8),
        track_hw=np.array([track_h, track_w], dtype=np.int32),
        original_hw=np.array(first_orig_hw, dtype=np.int32),
        resize_scale=np.array(scale, dtype=np.float32),
        bg_residual_px=bg_residual_px.astype(np.float32),
        bg_residual_score=bg_residual_score.astype(np.float32),
        per_frame_bg_residual=per_frame_bg_residual.astype(np.float32),
        bg_model_success=bg_model_success.astype(np.uint8),
        scene_flow_condition=scene_flow_condition.astype(np.uint8),
        visibility_condition=visibility_condition.astype(np.uint8),
        bg_condition=bg_condition.astype(np.uint8),
    )

    save_gray(out_dir / "cotracker_any4d_dynamic_probability.png", dynamic_prob)
    cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_mask.png"), dynamic_mask * 255)
    save_gray(out_dir / "cotracker_any4d_static_weight.png", static_weight_updated)
    cv2.imwrite(str(out_dir / "cotracker_any4d_static_ba_mask.png"), static_ba_mask * 255)
    cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_overlay.png"), overlay_prob(ref_bgr, dynamic_prob, (0, 0, 255), 0.45))
    cv2.imwrite(str(out_dir / "cotracker_any4d_static_ba_overlay.png"), overlay_prob(ref_bgr, static_ba_mask.astype(np.float32), (0, 255, 0), 0.35))
    cv2.imwrite(str(out_dir / "representative_points_overlay.png"), draw_points_overlay(ref_bgr, initial_points, dynamic_labels, point_dynamic_prob))
    cv2.imwrite(str(out_dir / "dynamic_tracks_overlay.png"), draw_track_overlay(ref_bgr, tracks, dynamic_labels))

    any4d_score_overlay = draw_score_overlay(
        ref_bgr,
        initial_points,
        point_motion_score,
        radius=args.point_vis_radius,
    )
    cv2.imwrite(
        str(out_dir / "point_any4d_motion_score_overlay.png"),
        any4d_score_overlay,
    )

    bg_residual_overlay = draw_score_overlay(
        ref_bgr,
        initial_points,
        bg_residual_score,
        radius=args.point_vis_radius,
    )
    cv2.imwrite(
        str(out_dir / "point_bg_residual_score_overlay.png"),
        bg_residual_overlay,
    )

    if scale != 1.0:
        orig_ref = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        prob_orig = resize_to(dynamic_prob, orig_ref.shape[:2], cv2.INTER_LINEAR)
        mask_orig = resize_to(dynamic_mask.astype(np.float32), orig_ref.shape[:2], cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_overlay_original_size.png"), overlay_prob(orig_ref, prob_orig, (0, 0, 255), 0.45))
        cv2.imwrite(str(out_dir / "cotracker_any4d_dynamic_mask_original_size.png"), (mask_orig > 0.5).astype(np.uint8) * 255)

    point_depth_norm = robust_normalize(point_depth, p_low=5.0, p_high=95.0)

    depth_overlay = draw_score_overlay(
        ref_bgr,
        initial_points,
        point_depth_norm,
        radius=args.point_vis_radius,
    )

    cv2.imwrite(
        str(out_dir / "point_vggt_depth_overlay.png"),
        depth_overlay,
    )

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
        "num_scene_flow_high_points": int(scene_flow_condition.sum()),
        "num_bg_residual_high_points": int(bg_condition.sum()),
        "mean_bg_residual_px": float(bg_residual_px.mean()),
        "mean_bg_residual_score": float(bg_residual_score.mean()),
        "use_bg_motion_residual": bool(args.use_bg_motion_residual),
        "bg_model": args.bg_model,
        "bg_ransac_thresh": args.bg_ransac_thresh,
        "bg_static_candidate_percentile": args.bg_static_candidate_percentile,
        "bg_min_points": args.bg_min_points,
        "bg_residual_score_thresh": args.bg_residual_score_thresh,
        "bg_residual_px_thresh": args.bg_residual_px_thresh,
        "bg_model_success_frames": int(bg_model_success.sum()),
        "use_depth_cc": bool(args.use_depth_cc),
        "cc_radius": args.cc_radius,
        "cc_depth_gate": args.cc_depth_gate,
        "cc_score_gate": args.cc_score_gate,
        "cc_min_component_points": args.cc_min_component_points,
        "cc_min_dynamic_points": args.cc_min_dynamic_points,
        "cc_min_dynamic_ratio": args.cc_min_dynamic_ratio,
        "cc_min_mean_score": args.cc_min_mean_score,
        "cc_propagate": bool(args.cc_propagate),
        "num_dynamic_points_before_cc": int(dynamic_labels_before_cc.sum()),
        "num_dynamic_points_after_cc": int(dynamic_labels.sum()),
        "num_valid_depth_cc_points": int(depth_cc_valid.sum()),
    }
    (out_dir / "cotracker_any4d_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[CoTrackerAny4D] Saved to {out_dir}")


if __name__ == "__main__":
    main()
