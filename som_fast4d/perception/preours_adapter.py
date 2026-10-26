"""Load actual saved Pre-Ours masks into Stage-1 PerceptionOutput.

This is a FILE-BASED real-output adapter, not a neural-network inference wrapper.
Original Pre-Ours inference must be executed in its own environment beforehand.
Pre-Ours outputs may use future frames; this adapter DOES NOT imply real-time or
causal operation. Frame names, shapes and mask polarity are validated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..core.data_types import FrameData, PerceptionOutput


def _resampling(name: str):
    modes = Image.Resampling if hasattr(Image, "Resampling") else Image
    return modes.NEAREST if name == "nearest" else modes.BILINEAR


def _load_mask(path: Path, hw, resize: bool) -> np.ndarray:
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        if raw.ndim != 2:
            raise ValueError("Mask must be [H,W]: %s" % path)
        if not np.isfinite(raw).all():
            raise ValueError("Mask has NaN/Inf: %s" % path)
        arr = np.asarray(raw, dtype=np.float32)
        threshold = 0.5 if float(np.max(arr)) <= 1.0 else 127.0
        binary = arr > threshold
        im = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")
    else:
        with Image.open(path) as src:
            im = src.convert("L").copy()
    if im.size != (hw[1], hw[0]):
        if not resize:
            raise ValueError("Mask resolution mismatch at %s: found %s expected %s. Use --resize-masks to explicitly resize." % (path, im.size, (hw[1], hw[0])))
        im = im.resize((hw[1], hw[0]), resample=_resampling("nearest"))
    return np.asarray(im) > 127


def _load_confidence(path: Path, hw, resize: bool) -> np.ndarray:
    x = np.load(path, allow_pickle=False)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Confidence must be finite [H,W]: %s" % path)
    x = np.asarray(x, dtype=np.float32)
    if x.shape != hw:
        if not resize:
            raise ValueError("Confidence resolution mismatch at %s: found %s expected %s" % (path, x.shape, hw))
        x = np.asarray(Image.fromarray(x, mode="F").resize((hw[1], hw[0]), resample=_resampling("bilinear")), dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


class PreOursPerceptionAdapter:
    """Load one actual Pre-Ours result per input frame.

    The adapter expects root/<RGB-stem>/ or root/<zero-padded-index>/ folders.
    Optional image-path metadata is checked to prevent mask/frame shifts.

    Mask semantics at boundary:
      dynamic_mask True = independent moving pixel (PNG 255).
      static_ba_mask True = confident static pixel and never dynamic.
      static_confidence in [0,1] from static_weight.
    """

    def __init__(self, preours_dir: str, frame_paths: List[str],
                 resize_masks: bool = False, static_weight_threshold: float = 0.45,
                 prefer_edge: bool = True, require_real_static_weight: bool = True):
        self.root = Path(preours_dir)
        self.frame_paths = [Path(p) for p in frame_paths]
        self.resize_masks = bool(resize_masks)
        self.static_weight_threshold = float(static_weight_threshold)
        self.prefer_edge = prefer_edge
        self.require_real_static_weight = require_real_static_weight
        if not self.frame_paths:
            raise ValueError("No RGB input paths")
        if len(set(p.name for p in self.frame_paths)) != len(self.frame_paths):
            raise ValueError("Duplicate RGB basenames; unable to align precomputed outputs")
        if not 0 <= self.static_weight_threshold <= 1:
            raise ValueError("static_weight_threshold must be between 0 and 1")
        self.last_source_dir = None

    def _dir_for(self, idx: int) -> Path:
        if idx < 0 or idx >= len(self.frame_paths):
            raise IndexError("Frame index outside configured source list")
        frame = self.frame_paths[idx]
        names = [frame.stem, str(idx), f"{idx:05d}", f"{idx:06d}", f"{idx:08d}"]
        found = list(dict.fromkeys(self.root / n for n in names if (self.root / n).is_dir()))
        if len(found) != 1:
            raise (FileNotFoundError if not found else ValueError)(
                "Expected exactly one Pre-Ours folder for %s index %d; candidates %s, found %s" % (frame.name, idx, names, found))
        return found[0]

    def _verify_alignment(self, frame_dir: Path, idx: int) -> None:
        expected = self.frame_paths[idx].name
        # refine_dynamic_mask.py exports summary.json with image_path.
        # edge_aware_refine.py exports edge_refine_summary.json with image_path.
        for filename in ("summary.json", "edge_refine_summary.json"):
            meta = frame_dir / filename
            if not meta.exists():
                continue
            values = json.loads(meta.read_text(encoding="utf8"))
            image_path = values.get("image_path")
            if image_path and Path(image_path).name != expected:
                raise ValueError("Pre-Ours frame alignment mismatch: expected %s but %s records %s" % (expected, meta, image_path))

    @staticmethod
    def _first(folder: Path, candidates: List[str]) -> Optional[Path]:
        return next((folder / n for n in candidates if (folder / n).exists()), None)

    def infer(self, frame: FrameData) -> PerceptionOutput:
        idx = frame.frame_index
        folder = self._dir_for(idx)
        self._verify_alignment(folder, idx)
        h, w = frame.depth_m.shape
        hw = (h, w)
        masks = (["edge_refined_dynamic_mask.png", "refined_dynamic_mask.png"] if self.prefer_edge
                 else ["refined_dynamic_mask.png", "edge_refined_dynamic_mask.png"])
        mask_file = self._first(folder, masks)
        if mask_file is None:
            raise FileNotFoundError("Missing real Pre-Ours refined_dynamic_mask.png in %s" % folder)
        dynamic = _load_mask(mask_file, hw, self.resize_masks)

        conf_names = (["edge_refined_static_weight.npy", "static_weight.npy"] if self.prefer_edge
                      else ["static_weight.npy", "edge_refined_static_weight.npy"])
        conf_file = self._first(folder, conf_names)
        if conf_file is None:
            if self.require_real_static_weight:
                raise FileNotFoundError("Missing real Pre-Ours static_weight.npy in %s" % folder)
            confidence = (~dynamic).astype(np.float32)
        else:
            confidence = _load_confidence(conf_file, hw, self.resize_masks)

        ba_names = (["edge_refined_static_ba_mask.npy", "static_ba_mask.png"] if self.prefer_edge
                    else ["static_ba_mask.png", "edge_refined_static_ba_mask.npy"])
        ba_file = self._first(folder, ba_names)
        static_ba = (_load_mask(ba_file, hw, self.resize_masks) if ba_file is not None
                     else confidence >= self.static_weight_threshold)

        # Crucial: never allow high-confidence dynamic pixels to drive camera tracking.
        static_ba = static_ba & ~dynamic & (confidence >= self.static_weight_threshold)
        static_ba &= np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        result = PerceptionOutput(dynamic_mask=dynamic.astype(np.bool_),
                                  static_confidence=confidence.astype(np.float32),
                                  static_ba_mask=static_ba.astype(np.bool_))
        result.validate(frame)
        self.last_source_dir = folder
        return result
