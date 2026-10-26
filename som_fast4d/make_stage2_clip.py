"""Copy an aligned, small RGB-D clip for Stage 2 without changing originals.

Run: python -m som_fast4d.make_stage2_clip --dataset DATASET --out data/stage2_clip --frames 6
Requires dataset/color/ and dataset/depth/ with matching basename stems.
The output uses 000000.png ... to enforce stable frame indexing.
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def make_clip(dataset: Path, out: Path, start: int = 0, count: int = 6):
    dataset, out = Path(dataset), Path(out)
    color_dir, depth_dir = dataset / "color", dataset / "depth"
    if start < 0 or count < 2:
        raise ValueError("start must be >=0 and count must be >=2")
    if out.resolve() == dataset.resolve() or dataset.resolve() in out.resolve().parents:
        raise ValueError("Output must not overlap the source dataset directory")
    if out.exists() and any(out.rglob("*")):
        raise FileExistsError("Output directory not empty: %s" % out)
    colors = sorted([p for p in color_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    depths = {p.stem: p for p in depth_dir.iterdir() if p.suffix.lower() in (".png", ".tiff", ".tif")}
    if start + count > len(colors):
        raise ValueError("Requested %d frames starting at %d, available %d" % (count, start, len(colors)))
    chosen = colors[start:start + count]
    missing = [p.name for p in chosen if p.stem not in depths]
    if missing:
        raise ValueError("RGB/depth stem mismatch: %s" % missing)
    (out / "color").mkdir(parents=True, exist_ok=False)
    (out / "depth").mkdir(parents=True, exist_ok=False)
    mapping = []
    for i, rgb_path in enumerate(chosen):
        depth_path = depths[rgb_path.stem]
        rgb_out = out / "color" / ("%06d.png" % i)
        depth_out = out / "depth" / ("%06d.png" % i)
        with Image.open(rgb_path) as im:
            im.convert("RGB").save(rgb_out)
        with Image.open(depth_path) as im:
            # Keep exact integer depth values when originally encoded in PNG.
            if depth_path.suffix.lower() == ".png":
                shutil.copyfile(depth_path, depth_out)
            else:
                im.save(depth_out)
        mapping.append({"stage2_index":i,"rgb_source":str(rgb_path),"depth_source":str(depth_path)})
    (out / "frame_manifest.json").write_text(json.dumps(mapping, indent=2), encoding="utf8")
    print("READY: %d aligned RGB-D frames at %s" % (len(chosen), out))
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Folder with color/ and depth/")
    p.add_argument("--out", default="data/stage2_bonn_6")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=6)
    a = p.parse_args()
    make_clip(Path(a.dataset), Path(a.out), a.start, a.frames)


if __name__ == "__main__":
    main()
