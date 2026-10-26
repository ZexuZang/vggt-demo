"""Run the repo's original edge_aware_refine.py over saved Pre-Ours results.

Requires real, already computed per-frame dynamic_probability.npy and
static_weight.npy; uses original OpenCV GrabCut implementation unchanged.
OpenCV must be installed in the selected Python environment.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--preours-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--allow-fallback", action="store_true", help="Skip frames where GrabCut finds no foreground; not recommended for final experiments")
    a = p.parse_args()
    images = sorted([p for p in Path(a.rgb_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    start, end = a.start, len(images) if a.max_frames is None else min(len(images), a.start+a.max_frames)
    if start < 0 or end <= start:
        raise ValueError("Invalid frame range")
    script = Path(__file__).resolve().parents[1] / "scripts/any4d_mask/edge_aware_refine.py"
    if not script.is_file():
        raise FileNotFoundError("Original Pre-Ours edge refiner is missing: " + str(script))
    for i in range(start, end):
        img = images[i]
        base = Path(a.preours_dir) / img.stem
        if not base.is_dir():
            raise FileNotFoundError("No Pre-Ours result for " + img.name)
        prob, weight = base / "dynamic_probability.npy", base / "static_weight.npy"
        if not (prob.is_file() and weight.is_file()):
            raise FileNotFoundError("Missing original Pre-Ours probability/weight for " + img.name)
        cmd=[a.python, str(script), "--image_path", str(img), "--prob_path", str(prob),
             "--static_weight_path", str(weight), "--output_dir", str(base)]
        print("[edge]", img.name, flush=True)
        if a.allow_fallback:
            res=subprocess.run(cmd,check=False)
            if res.returncode:
                print("[WARNING] edge refine failed for %s. Falling back to non-edge output." % img.name)
        else:
            subprocess.run(cmd, check=True)
    print("Stage 2 edge refinement attempt complete")


if __name__ == "__main__":
    main()
