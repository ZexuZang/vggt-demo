"""Offline pilot batch for per-frame Pre-Ours dynamic masks.

Uses the existing original Pre-Ours scripts unchanged. It REQUIRES an already
computed VGGT predictions .npz covering the SAME lexically sorted image list.
Default mode 'next' consumes a FUTURE frame, so outputs are NON-CAUSAL and may
only be used for offline mask-integration experiments, not realtime claims.

'past' mode avoids future image targets but has no past observation for frame 0.
Run with --start-index >= 1 and handle frame-0 initialization independently.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def run_cmd(cmd):
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", required=True, help="Images used for VGGT and Any4D, e.g. dataset/color")
    p.add_argument("--vggt-npz", required=True, help="Already saved whole-clip export_vggt_predictions output")
    p.add_argument("--any4d-root", required=True)
    p.add_argument("--any4d-checkpoint", required=True)
    p.add_argument("--out", required=True, help="Directory of numbered per-frame subfolders")
    p.add_argument("--any4d-python", default=sys.executable, help="Interpreter with Any4D installed")
    p.add_argument("--refine-python", default=sys.executable, help="Interpreter with numpy+opencv")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None, help="Exclusive; pilot 5 frames first")
    p.add_argument("--mode", choices=["next", "past"], default="next")
    a = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    image_dir = Path(a.image_dir).resolve()
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if len(images) < 2:
        raise ValueError("Need >=2 frames for Any4D scene motion estimation")
    with np.load(a.vggt_npz, allow_pickle=False) as vg:
        if "image_paths" not in vg:
            raise ValueError("VGGT npz missing image_paths")
        exported = [Path(str(s)).name for s in vg["image_paths"]]
        expected = [p.name for p in images]
        if exported != expected:
            raise ValueError("VGGT frame order does not match input image order")
    end = len(images) if a.end_index is None else min(len(images), a.end_index)
    if a.start_index < 0 or a.start_index >= end:
        raise ValueError("Invalid --start-index and --end-index")
    if a.mode == "past" and a.start_index == 0:
        raise ValueError("Past-only cannot produce frame 0 mask; start at index 1 and initialize first frame separately")

    for i in range(a.start_index, end):
        frame = images[i]
        if a.mode == "next" and i < len(images)-1:
            target = i + 1  # explicitly non-causal
        else:
            target = i - 1
        start, stop = target, target+1
        out_frame = Path(a.out) / frame.stem
        out_frame.mkdir(parents=True, exist_ok=True)
        any4d_pred = out_frame / "any4d_pred.npz"
        run_cmd([a.any4d_python, root / "scripts/any4d_mask/export_any4d_predictions.py",
                 "--any4d_root", a.any4d_root, "--image_dir", image_dir,
                 "--checkpoint_path", a.any4d_checkpoint,
                 "--ref_img_idx", i, "--start_idx", start, "--end_idx", stop,
                 "--output_npz", any4d_pred])
        run_cmd([a.refine_python, root / "scripts/any4d_mask/refine_dynamic_mask.py",
                 "--vggt_npz", a.vggt_npz, "--any4d_npz", any4d_pred,
                 "--image_dir", image_dir, "--ref_idx", i,
                 "--output_dir", out_frame])
        print("OK frame {}: {} (target {}, {} mode)".format(i, frame.name, target, a.mode))
    print("Per-frame masks ready at", a.out)
    if a.mode == "next":
        print("WARNING: NON-CAUSAL masks used future t+1 images; offline validation only.")


if __name__ == "__main__":
    main()
