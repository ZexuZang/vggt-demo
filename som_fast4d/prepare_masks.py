"""Convert per-frame Pre-Ours outputs into Flow4DGS-SLAM mask_colour images.

Flow4DGS-SLAM's realDataset expects ``mask_colour/*.png`` to store
255 for DYNAMIC pixels. It returns the logical inverse as motion_mask,
which is True for STATIC pixels eligible for tracking.

This script DOES NOT run VGGT or Any4D inference. Every frame must already
have its own Pre-Ours output, and future-frame use must be audited separately.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def _frame_dir(root, frame, index):
    names = [frame.stem, frame.name, str(index), "{:05d}".format(index),
             "{:06d}".format(index), "{:08d}".format(index)]
    found = [root / name for name in dict.fromkeys(names) if (root / name).is_dir()]
    if not found:
        raise FileNotFoundError("No Pre-Ours directory for frame {} (index {}). Checked {}".format(
            frame.name, index, names))
    if len(found) > 1:
        raise RuntimeError("Ambiguous frame directory for {}: {}".format(frame.name, found))
    return found[0]


def run(dataset, preours, output, prefer_edge=True, overwrite=False):
    dataset, preours, output = map(Path, (dataset, preours, output))
    images = sorted((dataset / "color").glob("*.png"), key=_natural_key)
    if not images:
        raise FileNotFoundError("No PNG frames under {}".format(dataset / "color"))
    if output.exists() and any(output.glob("*.png")) and not overwrite:
        raise FileExistsError("{} already has masks. Use --overwrite explicitly.".format(output))
    tasks = []
    for i, frame in enumerate(images):
        src_dir = _frame_dir(preours, frame, i)
        candidates = (["edge_refined_dynamic_mask.png", "refined_dynamic_mask.png"]
                      if prefer_edge else ["refined_dynamic_mask.png", "edge_refined_dynamic_mask.png"])
        src = next((src_dir / n for n in candidates if (src_dir / n).exists()), None)
        if src is None:
            raise FileNotFoundError("Missing refined dynamic mask in {}".format(src_dir))
        # If the original mask script wrote a frame path, protect against
        # accidental cross-frame alignment in sliding windows.
        summary_file = src_dir / "summary.json"
        if summary_file.is_file():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            recorded = summary.get("image_path")
            if recorded and Path(recorded).name != frame.name:
                raise ValueError("Frame alignment mismatch: {} vs {}".format(recorded, frame.name))
        tasks.append((frame, src, src_dir))

    output.mkdir(parents=True, exist_ok=True)
    confidence_dir = output.parent / "preours_static_confidence"
    confidence_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for frame, src, src_dir in tasks:
        with Image.open(frame) as rgb:
            size = rgb.size
        with Image.open(src) as mask_img:
            mask = mask_img.convert("L")
            if mask.size != size:
                mask = mask.resize(size, Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
            arr = np.asarray(mask)
        binary = (arr > 127).astype(np.uint8) * 255  # 255=dynamic
        Image.fromarray(binary, mode="L").save(output / frame.name)
        conf_path = src_dir / "static_weight.npy"
        if conf_path.is_file():
            conf = np.load(str(conf_path), allow_pickle=False)
            if conf.ndim != 2:
                raise ValueError("static_weight must be a 2D float map: {}".format(conf_path))
            conf_resized = Image.fromarray(conf.astype("float32"), mode="F").resize(
                size, Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
            np.save(str(confidence_dir / (frame.stem + ".npy")), np.asarray(conf_resized).clip(0, 1))
        report.append({"frame": frame.name, "source": str(src),
                       "dynamic_fraction": float((binary > 0).mean())})
    (output.parent / "preours_mask_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("OK: wrote {} dynamic masks to {}".format(len(report), output))
    print("IMPORTANT: Mask pixel 255 = dynamic; Flow4DGS motion_mask True = static.")
    print("Check future-frame leakage and processing latency before claiming online performance.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Flow4DGS dataset directory with color/*.png")
    parser.add_argument("--preours", required=True, help="Directory containing a subdirectory for each frame")
    parser.add_argument("--output", required=True, help="Usually <dataset>/mask_colour")
    parser.add_argument("--no-edge", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    a = parser.parse_args()
    run(a.dataset, a.preours, a.output, not a.no_edge, a.overwrite)


if __name__ == "__main__":
    main()
