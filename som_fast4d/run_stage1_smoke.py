"""Runnable CPU-only mock of the future SOM Fast4D time loop.

Run: python -m som_fast4d.run_stage1_smoke
All model outputs are SYNTHETIC. This does not run VGGT, Any4D,
Flow4DGS-SLAM, GenFlow3D, or evaluate their performance.
"""
import argparse
from typing import Optional, Sequence

import numpy as np

from .core import (CameraEstimate, Fast4DPipeline, FrameData,
                   GaussianMapState, MotionPrediction, PerceptionOutput)


class MockPerception:
    def infer(self, frame: FrameData) -> PerceptionOutput:
        # A bright square marks a synthetic dynamic object, for interface testing.
        dynamic = frame.rgb[..., 0] > 180
        static_ba = ~dynamic
        confidence = np.where(dynamic, 0.05, 0.95).astype(np.float32)
        return PerceptionOutput(dynamic, confidence, static_ba)


class MockTracking:
    def track(self, frame: FrameData, perception: PerceptionOutput,
              previous_map: Optional[GaussianMapState]) -> CameraEstimate:
        if not perception.tracking_mask(frame.depth_m).any():
            raise RuntimeError("No valid static pixels")
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = frame.frame_index * 0.01
        return CameraEstimate(T_wc)


class MockMapping:
    def __init__(self) -> None:
        self.last_prior_count = 0
        self.last_correction_m = 0.0

    def update(self, frame: FrameData, perception: PerceptionOutput,
               camera: CameraEstimate, previous_map: Optional[GaussianMapState],
               prior_for_current_frame: Optional[MotionPrediction]) -> GaussianMapState:
        self.last_prior_count = 0
        self.last_correction_m = 0.0
        if previous_map is None:
            centers = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32)
            ids = np.array([101, 202], dtype=np.int64)
            is_dynamic = np.array([False, True])
        else:
            centers = previous_map.centers_world_m.copy()
            ids = previous_map.gaussian_ids.copy()
            is_dynamic = previous_map.is_dynamic.copy()

        # Prediction is an INITIAL VALUE applied BEFORE the target observation.
        if prior_for_current_frame is not None:
            lookup = {int(v): i for i, v in enumerate(ids)}
            for gid, flow, valid in zip(prior_for_current_frame.gaussian_ids,
                                        prior_for_current_frame.delta_world_m,
                                        prior_for_current_frame.valid):
                if valid and int(gid) in lookup:
                    centers[lookup[int(gid)]] += flow
                    self.last_prior_count += 1

        # Synthetic measurement correction, stands in for observed flow / RGB-D.
        measured_x = 1.0 + 0.1 * frame.frame_index
        dynamic_rows = np.where(is_dynamic)[0]
        if len(dynamic_rows):
            self.last_correction_m = float(np.abs(centers[dynamic_rows, 0] - measured_x).mean())
            centers[dynamic_rows, 0] = measured_x
        return GaussianMapState(frame.frame_index, ids, centers, is_dynamic)


class MockPredictor:
    def predict_next(self, history: Sequence[GaussianMapState]) -> Optional[MotionPrediction]:
        if len(history) < 2:
            return None
        previous, current = history[-2], history[-1]
        prev_lookup = {int(gid): j for j, gid in enumerate(previous.gaussian_ids)}
        dynamic_rows = np.where(current.is_dynamic)[0]
        valid_rows = [i for i in dynamic_rows if int(current.gaussian_ids[i]) in prev_lookup]
        ids = current.gaussian_ids[valid_rows].copy()
        delta = np.stack([
            current.centers_world_m[i] - previous.centers_world_m[prev_lookup[int(current.gaussian_ids[i])]]
            for i in valid_rows
        ]).astype(np.float32) if valid_rows else np.zeros((0, 3), dtype=np.float32)
        return MotionPrediction(current.frame_index, current.frame_index + 1,
                                ids, delta, np.ones(len(ids), dtype=np.bool_))


def synthetic_frame(i: int, height: int = 32, width: int = 48) -> FrameData:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[12:23, 5 + i:16 + i, 0] = 240
    depth = np.full((height, width), 2.0, dtype=np.float32)
    K = np.array([[55., 0., width / 2.], [0., 55., height / 2.], [0., 0., 1.]])
    return FrameData(i, i / 30.0, rgb, depth, K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be >= 3 to demonstrate prediction correction")
    mapping = MockMapping()
    pipeline = Fast4DPipeline(MockPerception(), MockTracking(), mapping, MockPredictor())
    print("SYNTHETIC CPU SMOKE TEST - no real models are used")
    for i in range(args.frames):
        r = pipeline.process_frame(synthetic_frame(i))
        print("frame=%d static_pixels=%d prior_used=%d correction_m=%.4f prediction_for_t+1=%s" %
              (i, int(r.perception.tracking_mask(synthetic_frame(i).depth_m).sum()),
               mapping.last_prior_count, mapping.last_correction_m,
               "yes" if r.next_frame_prediction is not None else "no"))
    assert mapping.last_prior_count > 0
    print("PASS: causal prediction -> next-frame measurement correction contract")


if __name__ == "__main__":
    main()
