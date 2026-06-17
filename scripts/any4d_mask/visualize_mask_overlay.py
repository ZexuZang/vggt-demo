#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image_path)

    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask_path)

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    m = (mask.astype(np.float32) / 255.0)[..., None]
    red = np.zeros_like(img, dtype=np.float32)
    red[..., 2] = 255

    out = img.astype(np.float32) * (1 - args.alpha * m) + red * (args.alpha * m)
    out = np.clip(out, 0, 255).astype(np.uint8)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
