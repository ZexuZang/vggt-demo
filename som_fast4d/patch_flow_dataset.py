"""Apply a minimal idempotent bug-guard to Flow4DGS-SLAM's realDataset.

The original sorts color/depth numerically but masks lexicographically.
This can silently misalign frame masks. We only insert a numerical sort and
an alignment length check, backing up the unmodified .py file.
"""
import argparse
import shutil
from pathlib import Path


def patch(path):
    path = Path(path)
    src = path.read_text(encoding="utf-8")
    anchor = '        self.depth_paths = sorted(self.depth_paths, key=self.extract_number)\n'
    insert = '        self.mask_paths = sorted(self.mask_paths, key=self.extract_number)\n'
    if insert not in src:
        if src.count(anchor) != 1:
            raise RuntimeError("Could not uniquely find numeric depth-sort anchor; inspect repository version.")
        src = src.replace(anchor, anchor + insert)
    anchor2 = '        self.num_imgs = len(self.color_paths)\n        \n        self.dynamic = config["model_params"]["dynamic_model"]'
    check = ('        if self.mask_paths and len(self.mask_paths) != len(self.color_paths):\n'
             '            raise ValueError("mask_colour and color frame counts differ after slicing")\n')
    if check not in src:
        if src.count(anchor2) != 1:
            raise RuntimeError("Could not uniquely find post-slice anchor; inspect repository version.")
        src = src.replace(anchor2, check + anchor2)
    backup = path.with_suffix(path.suffix + ".pre_som_backup")
    if not backup.exists():
        shutil.copy2(str(path), str(backup))
    path.write_text(src, encoding="utf-8")
    print("Patched {} (backup: {})".format(path, backup))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", required=True, help="Path to Flow4DGS-SLAM/utils/dataset.py")
    args = p.parse_args()
    patch(args.file)
