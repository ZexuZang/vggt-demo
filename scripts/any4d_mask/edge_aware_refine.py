#!/usr/bin/env python3
"""
RGB edge-aware refinement for Any4D/VGGT dynamic masks.

This module refines a coarse Any4D/VGGT dynamic probability map with RGB
image boundaries using OpenCV GrabCut.

Inputs:
  - RGB frame folder or reference image
  - outputs/refined_mask/dynamic_probability.npy
  - optional outputs/refined_mask/static_weight.npy

Outputs:
  - edge_refined_dynamic_mask.png
  - edge_refined_dynamic_overlay.png
  - edge_refined_static_weight.npy/png
  - edge_refined_static_ba_mask.png
  - edge_refine_summary.json
"""
import argparse
from pathlib import Path
import json
import numpy as np
import cv2


def sorted_images(image_dir: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for e in exts:
        paths.extend(image_dir.glob(e))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def load_image(image_dir=None, image_path=None, ref_idx=0):
    if image_path:
        p = Path(image_path)
    else:
        paths = sorted_images(Path(image_dir))
        p = paths[ref_idx]
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img, p


def load_prob(path):
    p = Path(path)
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


def save_gray(path, x):
    x = np.clip(x, 0, 1)
    cv2.imwrite(str(path), (x * 255).astype(np.uint8))


def overlay(img_bgr, mask, color_bgr=(0, 0, 255), alpha=0.45):
    m = mask.astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    m = np.clip(m, 0, 1)
    if m.shape[:2] != img_bgr.shape[:2]:
        m = resize_to(m, img_bgr.shape[:2])
    color = np.zeros_like(img_bgr, dtype=np.float32)
    color[:, :] = np.array(color_bgr, dtype=np.float32)
    out = img_bgr.astype(np.float32) * (1 - alpha * m[..., None]) + color * (alpha * m[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def make_grabcut_seed(prob, fg_thresh, pr_fg_thresh, bg_thresh, dilate_px):
    seed = np.full(prob.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    seed[prob >= fg_thresh] = cv2.GC_FGD
    seed[(prob >= pr_fg_thresh) & (prob < fg_thresh)] = cv2.GC_PR_FGD
    seed[prob <= bg_thresh] = cv2.GC_BGD

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        prior = (prob >= pr_fg_thresh).astype(np.uint8)
        prior = cv2.dilate(prior, k, iterations=1)
        seed[(prior == 1) & (seed != cv2.GC_FGD) & (seed != cv2.GC_BGD)] = cv2.GC_PR_FGD
    return seed


def keep_motion_consistent_components(mask, prob, min_area=300, overlap_thresh=0.03):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    prior = prob > 0.25
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = labels == i
        overlap = (comp & prior).sum() / (comp.sum() + 1e-6)
        if overlap >= overlap_thresh:
            kept[comp] = 1
    return kept


def morph(mask, open_px=1, close_px=0):
    out = mask.astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--prob_path", required=True)
    parser.add_argument("--static_weight_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ref_idx", type=int, default=0)

    parser.add_argument("--fg_thresh", type=float, default=0.75)
    parser.add_argument("--pr_fg_thresh", type=float, default=0.35)
    parser.add_argument("--bg_thresh", type=float, default=0.10)
    parser.add_argument("--dilate_px", type=int, default=7)
    parser.add_argument("--grabcut_iters", type=int, default=5)

    parser.add_argument("--min_area", type=int, default=300)
    parser.add_argument("--component_overlap_thresh", type=float, default=0.03)
    parser.add_argument("--open_px", type=int, default=1)
    parser.add_argument("--close_px", type=int, default=0)

    parser.add_argument("--static_weight_thresh", type=float, default=0.45)
    parser.add_argument("--hard_remove_dynamic", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr, img_path = load_image(args.image_dir, args.image_path, args.ref_idx)
    h, w = img_bgr.shape[:2]

    prob = load_prob(args.prob_path)
    prob = resize_to(prob, (h, w), cv2.INTER_LINEAR)

    seed = make_grabcut_seed(
        prob,
        args.fg_thresh,
        args.pr_fg_thresh,
        args.bg_thresh,
        args.dilate_px,
    )

    if not np.any(seed == cv2.GC_FGD):
        raise RuntimeError("No sure foreground seed. Try lowering --fg_thresh to 0.65.")

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr,
        seed,
        None,
        bgd_model,
        fgd_model,
        args.grabcut_iters,
        cv2.GC_INIT_WITH_MASK,
    )

    refined = np.where(
        (seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)

    refined = keep_motion_consistent_components(
        refined,
        prob,
        min_area=args.min_area,
        overlap_thresh=args.component_overlap_thresh,
    )
    refined = morph(refined, open_px=args.open_px, close_px=args.close_px)

    if args.static_weight_path:
        static_weight = np.load(args.static_weight_path).astype(np.float32)
        static_weight = resize_to(static_weight, (h, w), cv2.INTER_LINEAR)
        static_weight = np.clip(static_weight, 0, 1)
    else:
        static_weight = 1.0 - prob

    if args.hard_remove_dynamic:
        static_weight_edge = static_weight.copy()
        static_weight_edge[refined > 0] = 0.0
    else:
        static_weight_edge = static_weight * (1.0 - refined.astype(np.float32))

    static_ba_mask_edge = (static_weight_edge > args.static_weight_thresh).astype(np.uint8)

    np.save(out_dir / "edge_refined_dynamic_mask.npy", refined.astype(np.uint8))
    np.save(out_dir / "edge_refined_static_weight.npy", static_weight_edge.astype(np.float32))
    np.save(out_dir / "edge_refined_static_ba_mask.npy", static_ba_mask_edge.astype(np.uint8))
    np.save(out_dir / "grabcut_seed.npy", seed.astype(np.uint8))

    save_gray(out_dir / "input_dynamic_probability_resized.png", prob)
    cv2.imwrite(str(out_dir / "edge_refined_dynamic_mask.png"), refined * 255)
    save_gray(out_dir / "edge_refined_static_weight.png", static_weight_edge)
    cv2.imwrite(str(out_dir / "edge_refined_static_ba_mask.png"), static_ba_mask_edge * 255)

    cv2.imwrite(str(out_dir / "edge_refined_dynamic_overlay.png"), overlay(img_bgr, refined, (0, 0, 255), 0.45))
    cv2.imwrite(str(out_dir / "edge_refined_static_ba_overlay.png"), overlay(img_bgr, static_ba_mask_edge, (0, 255, 0), 0.35))

    seed_vis = np.zeros((h, w, 3), dtype=np.uint8)
    seed_vis[seed == cv2.GC_BGD] = (0, 0, 0)
    seed_vis[seed == cv2.GC_PR_BGD] = (80, 80, 80)
    seed_vis[seed == cv2.GC_PR_FGD] = (0, 180, 255)
    seed_vis[seed == cv2.GC_FGD] = (0, 0, 255)
    cv2.imwrite(str(out_dir / "grabcut_seed_visualization.png"), seed_vis)

    summary = {
        "image_path": str(img_path),
        "prob_path": str(args.prob_path),
        "static_weight_path": str(args.static_weight_path) if args.static_weight_path else None,
        "output_dir": str(out_dir),
        "image_hw": [h, w],
        "fg_thresh": args.fg_thresh,
        "pr_fg_thresh": args.pr_fg_thresh,
        "bg_thresh": args.bg_thresh,
        "dilate_px": args.dilate_px,
        "grabcut_iters": args.grabcut_iters,
        "min_area": args.min_area,
        "open_px": args.open_px,
        "close_px": args.close_px,
        "dynamic_ratio_before": float((prob > 0.55).mean()),
        "dynamic_ratio_after": float(refined.mean()),
        "static_ba_ratio_after": float(static_ba_mask_edge.mean()),
    }
    (out_dir / "edge_refine_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[EdgeRefine] Saved to {out_dir}")


if __name__ == "__main__":
    main()
