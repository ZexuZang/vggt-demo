#!/usr/bin/env python3
"""
Patch the current vggt-demo DROID-W integration so external BA weights may be:
    [H,W]   scalar/isotropic (Baseline 1), or
    [2,H,W] x/y anisotropic precision (Ours).

The current repository already supports a scalar external static weight. This
script makes the smallest Python-side change needed for the Ours ablation:

1. datasets.py loads scalar or 2-channel .npy files.
2. depth_video.py stores two channels per keyframe.
3. depth_video.py multiplies DROID-W's learned x/y BA weights independently.

Backups are written next to modified files with suffix .bak_before_anisotropic.
Run this once after cloning/pulling the repository.
"""

from __future__ import annotations

import argparse
import py_compile
import re
from pathlib import Path


NEW_GET_WEIGHT = '''    def get_ours_static_weight(self, index):
        """
        Load external BA precision for frame index.

        Supported .npy shapes:
            [H,W]   -> isotropic scalar weight, repeated for x/y
            [2,H,W] -> anisotropic x/y precision
            [H,W,2] -> anisotropic x/y precision

        Return:
            torch.Tensor [2, H_out/8, W_out/8]
        """
        if not getattr(self, "use_ours_static_weight", False):
            return None

        if self.ours_static_weight_sequential:
            weight_path = os.path.join(
                self.ours_static_weight_dir,
                f"{index:06d}.npy",
            )
        else:
            img_stem = os.path.splitext(os.path.basename(self.color_paths[index]))[0]
            weight_path = os.path.join(
                self.ours_static_weight_dir,
                f"{img_stem}.npy",
            )

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"[OURS] BA weight not found: {weight_path}")

        weight = np.load(weight_path).astype(np.float32)
        weight = np.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=0.0)
        weight = np.clip(weight, 0.0, 1.0)

        if weight.ndim == 2:
            weight = np.stack([weight, weight], axis=-1)  # H,W,2
        elif weight.ndim == 3 and weight.shape[0] == 2:
            weight = np.transpose(weight, (1, 2, 0))      # H,W,2
        elif weight.ndim == 3 and weight.shape[-1] == 2:
            pass
        else:
            raise ValueError(
                f"[OURS] Expected [H,W], [2,H,W], or [H,W,2], got {weight.shape}"
            )

        weight = cv2.resize(
            weight,
            (self.W_out_with_edge, self.H_out_with_edge),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.W_edge > 0:
            edge = self.W_edge
            weight = weight[:, edge:-edge, :]
        if self.H_edge > 0:
            edge = self.H_edge
            weight = weight[edge:-edge, :, :]

        weight_t = torch.from_numpy(weight).float().permute(2, 0, 1)[None]
        weight_small = F.interpolate(
            weight_t,
            size=(self.H_out // 8, self.W_out // 8),
            mode="bilinear",
            align_corners=False,
        )[0]

        return torch.clamp(weight_small, 0.0, 1.0)

'''


NEW_STORAGE_BLOCK = '''        # ------------------------------------------------------------
        # Ours: store 2-channel external BA precision for this keyframe.
        # item[10] may be [H,W], [2,H,W], or [H,W,2].
        # ------------------------------------------------------------
        if len(item) > 10 and item[10] is not None:
            ours_w = item[10].float().to(self.device)

            if ours_w.ndim == 2:
                ours_w = ours_w[None].repeat(2, 1, 1)
            elif ours_w.ndim == 3 and ours_w.shape[0] == 1:
                ours_w = ours_w.repeat(2, 1, 1)
            elif ours_w.ndim == 3 and ours_w.shape[-1] == 2 and ours_w.shape[0] != 2:
                ours_w = ours_w.permute(2, 0, 1)
            elif ours_w.ndim == 3 and ours_w.shape[0] == 2:
                pass
            else:
                raise ValueError(
                    f"[OURS] Unsupported external BA weight shape: {tuple(ours_w.shape)}"
                )

            if ours_w.shape[-2:] != self.ours_static_weights[index].shape[-2:]:
                ours_w = F.interpolate(
                    ours_w[None],
                    size=self.ours_static_weights[index].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]

            self.ours_static_weights[index] = torch.clamp(ours_w, 0.0, 1.0)
        else:
            self.ours_static_weights[index].fill_(1.0)

'''


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak_before_anisotropic")
    if not backup_path.exists():
        backup_path.write_text(path.read_text())
        print(f"[BACKUP] {backup_path}")


def patch_datasets(path: Path) -> None:
    text = path.read_text()
    if "Supported .npy shapes:" in text:
        print(f"[SKIP] Already patched: {path}")
        return

    start = text.find("    def get_ours_static_weight(self, index):")
    end = text.find("    def get_intrinsic(self):", start)
    if start < 0 or end < 0:
        raise RuntimeError("Cannot locate get_ours_static_weight() in datasets.py")

    backup(path)
    path.write_text(text[:start] + NEW_GET_WEIGHT + text[end:])
    print(f"[PATCH] {path}")


def patch_depth_video(path: Path) -> None:
    text = path.read_text()
    if "Shape: [buffer, 2, ht/8, wd/8]" in text:
        print(f"[SKIP] Already patched: {path}")
        return

    backup(path)

    # 1. Update explanatory comment.
    text = text.replace(
        "# Shape: [buffer, ht/8, wd/8]",
        "# Shape: [buffer, 2, ht/8, wd/8]",
        1,
    )

    # 2. Allocate two x/y channels.
    pattern = re.compile(
        r"self\.ours_static_weights = torch\.ones\(\s*"
        r"buffer,\s*"
        r"ht // self\.down_scale,\s*"
        r"wd // self\.down_scale,\s*"
        r"device=self\.device,\s*"
        r"dtype=torch\.float,\s*"
        r"\)\.share_memory_\(\)",
        re.S,
    )
    replacement = '''self.ours_static_weights = torch.ones(
            buffer,
            2,
            ht // self.down_scale,
            wd // self.down_scale,
            device=self.device,
            dtype=torch.float,
        ).share_memory_()'''
    text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise RuntimeError("Cannot patch ours_static_weights allocation in depth_video.py")

    # 3. Replace scalar storage block.
    storage_start = text.find(
        "        # ------------------------------------------------------------\n"
        "        # Ours: store external static weight for this keyframe"
    )
    storage_end = text.find("    def __setitem__(self, index, item):", storage_start)
    if storage_start < 0 or storage_end < 0:
        raise RuntimeError("Cannot locate external-weight storage block in depth_video.py")
    text = text[:storage_start] + NEW_STORAGE_BLOCK + text[storage_end:]

    # 4. Replace scalar broadcast multiplication with direct channel-wise multiplication.
    old = "weight = weight * src_static_w[:, None, :, :]"
    new = "weight = weight * src_static_w"
    if old not in text:
        raise RuntimeError("Cannot locate scalar BA multiplication in depth_video.py")
    text = text.replace(old, new, 1)

    # Update nearby shape comment.
    text = text.replace(
        "# ours_static_weights[ii]: [E, H/8, W/8]",
        "# ours_static_weights[ii]: [E, 2, H/8, W/8]",
        1,
    )

    path.write_text(text)
    print(f"[PATCH] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, type=Path)
    args = parser.parse_args()

    droid_root = args.repo_root.resolve() / "third_party" / "DROID-W"
    datasets_py = droid_root / "src" / "utils" / "datasets.py"
    depth_video_py = droid_root / "src" / "depth_video.py"

    for path in (datasets_py, depth_video_py):
        if not path.exists():
            raise FileNotFoundError(path)

    patch_datasets(datasets_py)
    patch_depth_video(depth_video_py)

    py_compile.compile(str(datasets_py), doraise=True)
    py_compile.compile(str(depth_video_py), doraise=True)
    print("[OK] Python syntax check passed.")
    print("[NOTE] No CUDA extension rebuild is needed; this patch is Python-side only.")


if __name__ == "__main__":
    main()
