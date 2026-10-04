#!/usr/bin/env python3
"""
Generate the weight files needed for a three-way BA ablation in ZexuZang/vggt-demo.

Definitions used here
---------------------
Baseline 0
    Original DROID-W. No external static/dynamic weight is used.

Baseline 1
    Soft isotropic BA weighting from the refined dense dynamic probability:
        w_iso(x) = clip(1 - P_dyn(x), min_weight, 1)
    The same scalar weight is applied to the x/y reprojection residual channels.

Ours
    Motion-direction-aware diagonal Mahalanobis approximation. The same refined
    dense dynamic probability is used as the base reliability, while CoTracker
    trajectories are used only to estimate residual motion direction after
    camera/background motion compensation.

    Let g_i be the point motion residual after affine background compensation.
    Define z_i = ||g_i|| / sigma_g and
        kappa_i = kappa_max * z_i^2 / (1 + z_i^2)
        q_i = kappa_i / (1 + kappa_i)
    with e_i = g_i / ||g_i||. Then the diagonal precision is
        w_x = w_iso * (1 - q_i * e_x^2)
        w_y = w_iso * (1 - q_i * e_y^2)

Important
---------
This script does NOT rebuild P_dyn from sparse CoTracker points. P_dyn is loaded
from --refined_dynamic_path, consistent with the refined dense-mask workflow.
CoTracker tracks are used only for motion direction / camera compensation in Ours.

Expected input track file
-------------------------
outputs/.../cotracker_any4d_tracks.npz
with at least:
    tracks      [T, N, 2]
    visibility  [T, N]
    track_hw    [2]        (recommended)

Output
------
<output_dir>/baseline1/static_weight_npy/000000.npy ...  -> [H,W]
<output_dir>/ours/precision_weight_npy/000000.npy ...    -> [2,H,W]
<output_dir>/diagnostics/*.png
<output_dir>/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sorted_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in IMAGE_PATTERNS:
        paths.extend(image_dir.glob(pat))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def list_prob_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    files: list[Path] = []
    for pat in IMAGE_PATTERNS + ("*.npy",):
        files.extend(path.glob(pat))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No refined dynamic maps found in {path}")
    return files


def load_prob_map(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        x = np.load(path).astype(np.float32)
    else:
        x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if x is None:
            raise FileNotFoundError(path)
        x = x.astype(np.float32)
        if x.ndim == 3:
            x = x[..., 0]
        if x.max(initial=0.0) > 1.0:
            x /= 255.0
    x = np.squeeze(x)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    k = max(3, int(round(6.0 * sigma + 1.0)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(x.astype(np.float32), (k, k), sigma)


def resize_tracks_to_image(
    points: np.ndarray,
    track_hw: Optional[tuple[int, int]],
    image_hw: tuple[int, int],
) -> np.ndarray:
    pts = points.astype(np.float32).copy()
    if track_hw is None:
        return pts
    track_h, track_w = track_hw
    h, w = image_hw
    if (track_h, track_w) != (h, w):
        pts[:, 0] *= w / max(float(track_w), 1.0)
        pts[:, 1] *= h / max(float(track_h), 1.0)
    return pts


def bilinear_sample(map_hw: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    h, w = map_hw.shape[:2]
    x = np.clip(points_xy[:, 0], 0, w - 1).astype(np.float32)
    y = np.clip(points_xy[:, 1], 0, h - 1).astype(np.float32)
    sample = cv2.remap(
        map_hw.astype(np.float32),
        x.reshape(-1, 1),
        y.reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sample.reshape(-1).astype(np.float32)


def fit_background_affine(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    valid_mask: np.ndarray,
    ransac_thresh_px: float,
) -> Optional[np.ndarray]:
    ids = np.where(valid_mask)[0]
    if len(ids) < 6:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        prev_pts[ids].astype(np.float32),
        curr_pts[ids].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh_px),
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    return M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = np.concatenate(
        [pts.astype(np.float32), np.ones((len(pts), 1), dtype=np.float32)], axis=1
    )
    return (homo @ M.T).astype(np.float32)


def robust_sigma(values: np.ndarray, floor: float) -> float:
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float(floor)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return float(max(1.4826 * mad, floor))


def rasterize_direction_penalty(
    points_xy: np.ndarray,
    visibility_t: np.ndarray,
    residual_vec: np.ndarray,
    sigma_g: float,
    image_hw: tuple[int, int],
    point_radius: int,
    visibility_thresh: float,
    kappa_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the diagonal directional penalty q*e_x^2 and q*e_y^2."""
    h, w = image_hw
    qx_map = np.zeros((h, w), dtype=np.float32)
    qy_map = np.zeros((h, w), dtype=np.float32)

    mag = np.linalg.norm(residual_vec, axis=1)
    z = mag / max(float(sigma_g), 1e-6)
    kappa = float(kappa_max) * (z * z) / (1.0 + z * z)
    q = kappa / (1.0 + kappa)

    denom = np.maximum(mag, 1e-6)
    ex = residual_vec[:, 0] / denom
    ey = residual_vec[:, 1] / denom
    qx = q * ex * ex
    qy = q * ey * ey

    for i, ((x, y), vis) in enumerate(zip(points_xy, visibility_t)):
        if vis < visibility_thresh:
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        tmp = np.zeros((h, w), dtype=np.float32)
        cv2.circle(tmp, (xi, yi), int(point_radius), 1.0, -1, lineType=cv2.LINE_AA)
        mask = tmp > 0
        qx_map[mask] = np.maximum(qx_map[mask], float(qx[i]))
        qy_map[mask] = np.maximum(qy_map[mask], float(qy[i]))

    return np.clip(qx_map, 0.0, 0.999), np.clip(qy_map, 0.0, 0.999)


def save_gray(path: Path, x: np.ndarray) -> None:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    cv2.imwrite(str(path), (255.0 * x).astype(np.uint8))


def select_refined_file(
    refined_files: list[Path],
    frame_idx: int,
    repeat_last: bool,
) -> Path:
    if len(refined_files) == 1:
        return refined_files[0]
    if frame_idx < len(refined_files):
        return refined_files[frame_idx]
    if repeat_last:
        return refined_files[-1]
    raise RuntimeError(
        f"Frame {frame_idx}: only {len(refined_files)} refined maps available. "
        "Use --repeat_last_refined only if repeating the last map is intentional."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_npz", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--refined_dynamic_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)

    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--repeat_last_refined", action="store_true")
    parser.add_argument("--visibility_thresh", type=float, default=0.5)
    parser.add_argument("--min_weight", type=float, default=0.05)

    parser.add_argument("--static_seed_thresh", type=float, default=0.25)
    parser.add_argument("--ransac_thresh_px", type=float, default=2.0)
    parser.add_argument("--motion_sigma_floor_px", type=float, default=0.5)
    parser.add_argument("--kappa_max", type=float, default=12.0)
    parser.add_argument("--point_radius", type=int, default=4)
    parser.add_argument("--direction_blur_sigma", type=float, default=1.5)
    args = parser.parse_args()

    tracks_pack = np.load(args.tracks_npz, allow_pickle=True)
    for key in ("tracks", "visibility"):
        if key not in tracks_pack:
            raise KeyError(f"Missing '{key}' in {args.tracks_npz}")

    tracks = tracks_pack["tracks"].astype(np.float32)
    visibility = tracks_pack["visibility"].astype(np.float32)
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    if visibility.shape[:2] != tracks.shape[:2]:
        raise ValueError(
            f"visibility {visibility.shape} does not match tracks {tracks.shape}"
        )

    track_hw = None
    if "track_hw" in tracks_pack:
        track_hw = tuple(int(x) for x in tracks_pack["track_hw"].tolist())

    image_paths = sorted_images(args.image_dir)
    refined_files = list_prob_files(args.refined_dynamic_path)

    T = min(len(image_paths), tracks.shape[0])
    if args.max_frames is not None:
        T = min(T, args.max_frames)

    baseline1_dir = args.output_dir / "baseline1" / "static_weight_npy"
    ours_dir = args.output_dir / "ours" / "precision_weight_npy"
    diagnostic_dir = args.output_dir / "diagnostics"
    for d in (baseline1_dir, ours_dir, diagnostic_dir):
        ensure_dir(d)

    frame_stats: list[dict] = []

    for t in range(T):
        image = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_paths[t])
        h, w = image.shape[:2]

        refined_file = select_refined_file(refined_files, t, args.repeat_last_refined)
        p_dyn = load_prob_map(refined_file)
        p_dyn = cv2.resize(p_dyn, (w, h), interpolation=cv2.INTER_LINEAR)
        p_dyn = np.clip(p_dyn, 0.0, 1.0).astype(np.float32)

        # Baseline 1: direct refined dense soft weight.
        w_iso = np.clip(1.0 - p_dyn, args.min_weight, 1.0).astype(np.float32)

        curr_pts = resize_tracks_to_image(tracks[t], track_hw, (h, w))

        if t == 0:
            residual = np.zeros_like(curr_pts, dtype=np.float32)
            sigma_g = float(args.motion_sigma_floor_px)
            qx_map = np.zeros((h, w), dtype=np.float32)
            qy_map = np.zeros((h, w), dtype=np.float32)
            affine_ok = False
            num_static_seeds = 0
        else:
            prev_pts = resize_tracks_to_image(tracks[t - 1], track_hw, (h, w))
            visible_pair = (
                (visibility[t - 1] >= args.visibility_thresh)
                & (visibility[t] >= args.visibility_thresh)
            )

            # Static seeds come from the refined dense dynamic probability,
            # not from sparse point dynamic labels.
            point_pdyn = bilinear_sample(p_dyn, curr_pts)
            static_seed = visible_pair & (point_pdyn <= args.static_seed_thresh)
            num_static_seeds = int(static_seed.sum())

            M = fit_background_affine(
                prev_pts,
                curr_pts,
                static_seed,
                args.ransac_thresh_px,
            )
            if M is None:
                M = fit_background_affine(
                    prev_pts,
                    curr_pts,
                    visible_pair,
                    args.ransac_thresh_px,
                )

            if M is None:
                pred = prev_pts
                affine_ok = False
            else:
                pred = apply_affine(M, prev_pts)
                affine_ok = True

            residual = (curr_pts - pred).astype(np.float32)
            residual[~visible_pair] = 0.0

            sigma_mask = static_seed if static_seed.sum() >= 6 else visible_pair
            sigma_g = robust_sigma(
                np.linalg.norm(residual[sigma_mask], axis=1),
                floor=args.motion_sigma_floor_px,
            )

            qx_map, qy_map = rasterize_direction_penalty(
                points_xy=curr_pts,
                visibility_t=visibility[t],
                residual_vec=residual,
                sigma_g=sigma_g,
                image_hw=(h, w),
                point_radius=args.point_radius,
                visibility_thresh=args.visibility_thresh,
                kappa_max=args.kappa_max,
            )
            qx_map = np.clip(
                gaussian_blur(qx_map, args.direction_blur_sigma), 0.0, 0.999
            )
            qy_map = np.clip(
                gaussian_blur(qy_map, args.direction_blur_sigma), 0.0, 0.999
            )

        # Ours: diagonal approximation of a 2x2 Mahalanobis precision.
        w_x = np.clip(w_iso * (1.0 - qx_map), args.min_weight, 1.0).astype(np.float32)
        w_y = np.clip(w_iso * (1.0 - qy_map), args.min_weight, 1.0).astype(np.float32)
        w_xy = np.stack([w_x, w_y], axis=0)  # [2,H,W]

        stem = f"{t:06d}"
        np.save(baseline1_dir / f"{stem}.npy", w_iso)
        np.save(ours_dir / f"{stem}.npy", w_xy)

        save_gray(diagnostic_dir / f"{stem}_refined_dynamic.png", p_dyn)
        save_gray(diagnostic_dir / f"{stem}_baseline1_weight.png", w_iso)
        save_gray(diagnostic_dir / f"{stem}_ours_wx.png", w_x)
        save_gray(diagnostic_dir / f"{stem}_ours_wy.png", w_y)

        residual_mag = np.linalg.norm(residual, axis=1)
        frame_stats.append(
            {
                "frame": t,
                "image": str(image_paths[t]),
                "refined_dynamic": str(refined_file),
                "affine_ok": bool(affine_ok),
                "num_static_seeds": int(num_static_seeds),
                "motion_sigma_px": float(sigma_g),
                "residual_median_px": float(np.median(residual_mag)),
                "residual_p90_px": float(np.percentile(residual_mag, 90)),
                "dynamic_probability_mean": float(p_dyn.mean()),
                "baseline1_weight_mean": float(w_iso.mean()),
                "ours_wx_mean": float(w_x.mean()),
                "ours_wy_mean": float(w_y.mean()),
            }
        )

        print(
            f"[{t:04d}/{T - 1:04d}] "
            f"Pdyn={p_dyn.mean():.4f} "
            f"B1={w_iso.mean():.4f} "
            f"Ours(x/y)={w_x.mean():.4f}/{w_y.mean():.4f} "
            f"sigma={sigma_g:.3f}px"
        )

    summary = {
        "definitions": {
            "baseline0": "Original DROID-W, external weight disabled.",
            "baseline1": "Soft isotropic weight = clip(1 - refined_dynamic_probability, min_weight, 1).",
            "ours": "Baseline-1 reliability plus camera-compensated motion-direction diagonal Mahalanobis precision.",
        },
        "tracks_npz": str(args.tracks_npz),
        "image_dir": str(args.image_dir),
        "refined_dynamic_path": str(args.refined_dynamic_path),
        "num_frames": int(T),
        "track_hw": track_hw,
        "min_weight": args.min_weight,
        "static_seed_thresh": args.static_seed_thresh,
        "ransac_thresh_px": args.ransac_thresh_px,
        "motion_sigma_floor_px": args.motion_sigma_floor_px,
        "kappa_max": args.kappa_max,
        "baseline1_weight_dir": str(baseline1_dir),
        "ours_weight_dir": str(ours_dir),
        "frames": frame_stats,
    }
    ensure_dir(args.output_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[DONE]")
    print("Baseline 1:", baseline1_dir)
    print("Ours:      ", ours_dir)


if __name__ == "__main__":
    main()
