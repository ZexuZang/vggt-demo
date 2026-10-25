"""Shared numpy contracts for Pre-Ours, Flow4DGS-SLAM and GenFlow3D.

All model-specific tensor conversions should happen in adapters, not here.
Conventions:
  - RGB: uint8 HxWx3, RGB channel order.
  - Depth: float32 HxW, METERS, 0 or NaN denotes invalid measurement.
  - Masks: bool HxW; dynamic=True means *moving*; static_ba=True means *usable static*.
  - Poses: T_world_camera transforms points from camera to world, meters.
  - 3D coordinates/scene flow: world frame, meters / displacement per frame.
  - Time: timestamp_s in seconds, monotonically increasing.
  - Gaussian identity: integer IDs stable over time, not row positions.
These explicit contracts prevent subtle mask, coordinate and frame-index mix-ups.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _xyz(arr: np.ndarray, name: str) -> None:
    _require(isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3,
             "%s must be ndarray [N,3]" % name)
    _require(np.issubdtype(arr.dtype, np.floating), "%s must be floating point" % name)
    _require(np.isfinite(arr).all(), "%s contains NaN/Inf" % name)


@dataclass
class FrameData:
    frame_index: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray

    def __post_init__(self) -> None:
        _require(isinstance(self.frame_index, int) and self.frame_index >= 0,
                 "frame_index must be nonnegative int")
        _require(np.isfinite(self.timestamp_s), "timestamp_s must be finite")
        _require(isinstance(self.rgb, np.ndarray) and self.rgb.ndim == 3
                 and self.rgb.shape[2] == 3 and self.rgb.dtype == np.uint8,
                 "rgb must be uint8 [H,W,3] in RGB order")
        h, w = self.rgb.shape[:2]
        _require(isinstance(self.depth_m, np.ndarray) and self.depth_m.shape == (h, w)
                 and np.issubdtype(self.depth_m.dtype, np.floating),
                 "depth_m must be floating [H,W], meters")
        _require(isinstance(self.K, np.ndarray) and self.K.shape == (3, 3)
                 and np.isfinite(self.K).all(), "K must be finite [3,3]")
        _require(float(self.K[0, 0]) > 0 and float(self.K[1, 1]) > 0,
                 "K requires positive focal lengths")


@dataclass
class PerceptionOutput:
    # Explicit mask semantics; never silently invert at pipeline boundaries.
    dynamic_mask: np.ndarray
    static_confidence: np.ndarray
    static_ba_mask: np.ndarray

    def validate(self, frame: FrameData) -> None:
        shape = frame.depth_m.shape
        for name in ("dynamic_mask", "static_ba_mask"):
            x = getattr(self, name)
            _require(isinstance(x, np.ndarray) and x.shape == shape and x.dtype == np.bool_,
                     "%s must be bool [H,W] with True semantics documented" % name)
        x = self.static_confidence
        _require(isinstance(x, np.ndarray) and x.shape == shape
                 and np.issubdtype(x.dtype, np.floating),
                 "static_confidence must be floating [H,W]")
        _require(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all(),
                 "static_confidence must lie in [0,1]")
        _require(not (self.dynamic_mask & self.static_ba_mask).any(),
                 "static_ba_mask cannot include dynamic pixels")

    def tracking_mask(self, depth_m: np.ndarray) -> np.ndarray:
        """Only confident static pixels with valid positive depth."""
        return self.static_ba_mask & ~self.dynamic_mask & np.isfinite(depth_m) & (depth_m > 0)


@dataclass
class CameraEstimate:
    # Explicit direction avoids accidental T_cw vs T_wc inversion.
    T_world_camera: np.ndarray

    def __post_init__(self) -> None:
        T = self.T_world_camera
        _require(isinstance(T, np.ndarray) and T.shape == (4, 4) and np.isfinite(T).all(),
                 "T_world_camera must be finite [4,4]")
        _require(np.allclose(T[3], [0., 0., 0., 1.], atol=1e-5),
                 "pose matrix last row must be [0,0,0,1]")


@dataclass
class GaussianMapState:
    frame_index: int
    gaussian_ids: np.ndarray  # integer [N], unique, stable identities
    centers_world_m: np.ndarray  # float [N,3]
    is_dynamic: np.ndarray  # bool [N]; other attributes remain in Flow4DGS

    def __post_init__(self) -> None:
        _require(isinstance(self.frame_index, int) and self.frame_index >= 0,
                 "GaussianMapState.frame_index must be >= 0")
        _xyz(self.centers_world_m, "centers_world_m")
        n = len(self.centers_world_m)
        ids = self.gaussian_ids
        _require(isinstance(ids, np.ndarray) and ids.shape == (n,)
                 and np.issubdtype(ids.dtype, np.integer), "gaussian_ids must be integer [N]")
        _require(len(np.unique(ids)) == n, "gaussian_ids must be unique")
        _require(isinstance(self.is_dynamic, np.ndarray) and self.is_dynamic.shape == (n,)
                 and self.is_dynamic.dtype == np.bool_, "is_dynamic must be bool [N]")


@dataclass
class MotionPrediction:
    # Conditional estimate made at t for t+1; never includes the future frame.
    source_frame_index: int
    target_frame_index: int
    gaussian_ids: np.ndarray  # predict only matched existing dynamic Gaussians
    delta_world_m: np.ndarray  # [M,3] displacement, not velocity
    valid: np.ndarray  # [M] usable confidence flag
    uncertainty_std_m: Optional[np.ndarray] = None  # optional [M,3] positive stdev

    def __post_init__(self) -> None:
        _require(self.source_frame_index >= 0 and
                 self.target_frame_index == self.source_frame_index + 1,
                 "prediction must target the NEXT frame only")
        _xyz(self.delta_world_m, "delta_world_m")
        m = len(self.delta_world_m)
        _require(isinstance(self.gaussian_ids, np.ndarray)
                 and self.gaussian_ids.shape == (m,)
                 and np.issubdtype(self.gaussian_ids.dtype, np.integer)
                 and len(np.unique(self.gaussian_ids)) == m,
                 "prediction gaussian_ids must be unique integer [M]")
        _require(isinstance(self.valid, np.ndarray) and self.valid.shape == (m,)
                 and self.valid.dtype == np.bool_, "prediction valid must be bool [M]")
        if self.uncertainty_std_m is not None:
            s = self.uncertainty_std_m
            _require(isinstance(s, np.ndarray) and s.shape == (m, 3)
                     and np.isfinite(s).all() and (s > 0).all(),
                     "uncertainty_std_m must be positive finite [M,3]")

    def validate_source(self, source_map: GaussianMapState) -> None:
        _require(source_map.frame_index == self.source_frame_index,
                 "prediction source frame and source Gaussian map differ")
        dynamic_ids = set(map(int, source_map.gaussian_ids[source_map.is_dynamic]))
        _require(all(int(i) in dynamic_ids for i in self.gaussian_ids),
                 "predicted IDs must be dynamic Gaussians in source map")
