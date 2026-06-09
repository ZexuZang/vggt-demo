#!/usr/bin/env python3
"""
Run VGGT COLMAP demo in a reproducible way.

Recommended workflow:
1. Edit this file in VS Code.
2. Commit and push to GitHub.
3. In Colab, clone/pull the repo.
4. Run this script with paths pointing to Google Drive data.

Expected source data structure:
SOURCE_SCENE_DIR/
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...

Output:
OUTPUT_SCENE_DIR/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── points.ply
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy images to local Colab storage, run VGGT demo_colmap.py, and copy results back."
    )

    parser.add_argument(
        "--source_scene_dir",
        type=Path,
        required=True,
        help="Scene directory containing an images/ folder. In Colab this is usually under /content/drive/MyDrive/...",
    )
    parser.add_argument(
        "--local_scene_dir",
        type=Path,
        default=Path("/content/vggt_scene"),
        help="Temporary local scene directory on Colab. Running from local disk is usually faster than Drive.",
    )
    parser.add_argument(
        "--output_scene_dir",
        type=Path,
        default=None,
        help="Where to copy the final sparse/ result. If omitted, results are copied back to source_scene_dir.",
    )

    parser.add_argument(
        "--clean_local",
        action="store_true",
        help="Delete local_scene_dir before copying images. Recommended for a fresh run.",
    )
    parser.add_argument(
        "--skip_copy",
        action="store_true",
        help="Skip copying images from source_scene_dir to local_scene_dir. Use this if local_scene_dir is already prepared.",
    )

    # Common VGGT demo_colmap.py arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ba", action="store_true", help="Use bundle adjustment. Slower but may improve results.")
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)

    return parser.parse_args()


def get_repo_root() -> Path:
    """
    This script is expected to live in:
        REPO_ROOT/scripts/run_vggt_colmap.py

    Therefore, repo root is parent of parent.
    """
    return Path(__file__).resolve().parents[1]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find images folder: {image_dir}")

    image_paths = [
        p for p in sorted(image_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in {image_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    return image_paths


def prepare_local_scene(source_scene_dir: Path, local_scene_dir: Path, clean_local: bool, skip_copy: bool) -> None:
    source_images_dir = source_scene_dir / "images"
    local_images_dir = local_scene_dir / "images"

    source_images = list_images(source_images_dir)
    print(f"[INFO] Found {len(source_images)} images in: {source_images_dir}")

    if skip_copy:
        local_images = list_images(local_images_dir)
        print(f"[INFO] Skip copy enabled. Found {len(local_images)} local images in: {local_images_dir}")
        return

    if clean_local and local_scene_dir.exists():
        print(f"[INFO] Removing old local scene directory: {local_scene_dir}")
        shutil.rmtree(local_scene_dir)

    local_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Copying images to local directory: {local_images_dir}")
    for src in source_images:
        dst = local_images_dir / src.name
        shutil.copy2(src, dst)

    local_images = list_images(local_images_dir)
    print(f"[INFO] Prepared {len(local_images)} local images.")


def run_vggt_demo(repo_root: Path, local_scene_dir: Path, args: argparse.Namespace) -> None:
    demo_script = repo_root / "demo_colmap.py"
    if not demo_script.exists():
        raise FileNotFoundError(
            f"Cannot find {demo_script}. "
            "Please put this script inside the official VGGT repo, e.g. vggt/scripts/run_vggt_colmap.py"
        )

    cmd = [
        sys.executable,
        str(demo_script),
        "--scene_dir",
        str(local_scene_dir),
        "--seed",
        str(args.seed),
        "--conf_thres_value",
        str(args.conf_thres_value),
    ]

    if args.use_ba:
        cmd.extend([
            "--use_ba",
            "--max_query_pts",
            str(args.max_query_pts),
            "--query_frame_num",
            str(args.query_frame_num),
        ])

    print("[INFO] Running VGGT command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def copy_results_back(local_scene_dir: Path, output_scene_dir: Path) -> None:
    local_sparse_dir = local_scene_dir / "sparse"
    output_sparse_dir = output_scene_dir / "sparse"

    if not local_sparse_dir.exists():
        raise FileNotFoundError(f"VGGT finished, but sparse/ was not found at: {local_sparse_dir}")

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    if output_sparse_dir.exists():
        print(f"[INFO] Removing old output sparse directory: {output_sparse_dir}")
        shutil.rmtree(output_sparse_dir)

    print(f"[INFO] Copying results back to: {output_sparse_dir}")
    shutil.copytree(local_sparse_dir, output_sparse_dir)

    print("[INFO] Done. Output files:")
    for p in sorted(output_sparse_dir.iterdir()):
        print(f"  - {p}")


def main() -> None:
    args = parse_args()

    source_scene_dir = args.source_scene_dir.expanduser().resolve()
    local_scene_dir = args.local_scene_dir.expanduser().resolve()
    output_scene_dir = args.output_scene_dir
    if output_scene_dir is None:
        output_scene_dir = source_scene_dir
    else:
        output_scene_dir = output_scene_dir.expanduser().resolve()

    repo_root = get_repo_root()

    print(f"[INFO] Repo root:        {repo_root}")
    print(f"[INFO] Source scene dir: {source_scene_dir}")
    print(f"[INFO] Local scene dir:  {local_scene_dir}")
    print(f"[INFO] Output scene dir: {output_scene_dir}")

    prepare_local_scene(
        source_scene_dir=source_scene_dir,
        local_scene_dir=local_scene_dir,
        clean_local=args.clean_local,
        skip_copy=args.skip_copy,
    )

    run_vggt_demo(
        repo_root=repo_root,
        local_scene_dir=local_scene_dir,
        args=args,
    )

    copy_results_back(
        local_scene_dir=local_scene_dir,
        output_scene_dir=output_scene_dir,
    )


if __name__ == "__main__":
    main()
