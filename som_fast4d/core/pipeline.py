"""Single-process interface skeleton: perception -> tracking -> mapping -> prediction.

This module is a TESTABLE ORCHESTRATOR, NOT a full GPU SLAM implementation.
The predictor is called only AFTER current frame mapping; its result cannot
observe frame t+1 and is consumed by the mapper when t+1 arrives.
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .data_types import (CameraEstimate, FrameData, GaussianMapState,
                         MotionPrediction, PerceptionOutput)


class PerceptionAdapter(Protocol):
    def infer(self, frame: FrameData) -> PerceptionOutput:
        ...


class TrackingAdapter(Protocol):
    def track(self, frame: FrameData, perception: PerceptionOutput,
              previous_map: Optional[GaussianMapState]) -> CameraEstimate:
        ...


class MappingAdapter(Protocol):
    def update(self, frame: FrameData, perception: PerceptionOutput,
               camera: CameraEstimate, previous_map: Optional[GaussianMapState],
               prior_for_current_frame: Optional[MotionPrediction]) -> GaussianMapState:
        """Implement observation correction here; prior is only initialization."""
        ...


class PredictorAdapter(Protocol):
    def predict_next(self, history: Sequence[GaussianMapState]) -> Optional[MotionPrediction]:
        """Only history up to current frame, never target frame observations."""
        ...


@dataclass
class FrameResult:
    frame_index: int
    perception: PerceptionOutput
    camera: CameraEstimate
    gaussian_map: GaussianMapState
    prior_used_for_current_frame: Optional[MotionPrediction]
    next_frame_prediction: Optional[MotionPrediction]


class Fast4DPipeline:
    """Causally executes components and validates module-level contracts."""

    def __init__(self, perception: PerceptionAdapter, tracking: TrackingAdapter,
                 mapping: MappingAdapter, predictor: PredictorAdapter,
                 max_history: int = 4) -> None:
        if max_history < 2:
            raise ValueError("max_history must be >=2")
        self.perception = perception
        self.tracking = tracking
        self.mapping = mapping
        self.predictor = predictor
        self.max_history = max_history
        self.history: List[GaussianMapState] = []
        self.previous_timestamp_s: Optional[float] = None
        self.pending_prediction: Optional[MotionPrediction] = None

    def process_frame(self, frame: FrameData) -> FrameResult:
        if self.history:
            last = self.history[-1]
            if frame.frame_index != last.frame_index + 1:
                raise ValueError("Frames must be consecutive: expected %d got %d" %
                                 (last.frame_index + 1, frame.frame_index))
            if self.previous_timestamp_s is not None and frame.timestamp_s <= self.previous_timestamp_s:
                raise ValueError("Frame timestamps must increase")
        perception = self.perception.infer(frame)
        perception.validate(frame)
        old_map = self.history[-1] if self.history else None
        # Use only the prediction produced at the PREVIOUS frame.
        prior = self.pending_prediction
        if prior is not None:
            if old_map is None or prior.target_frame_index != frame.frame_index:
                raise ValueError("Prediction/frame misalignment")
            prior.validate_source(old_map)
        camera = self.tracking.track(frame, perception, old_map)
        updated = self.mapping.update(frame, perception, camera, old_map, prior)
        if updated.frame_index != frame.frame_index:
            raise ValueError("Mapping output frame index mismatch")
        self.history.append(updated)
        self.history = self.history[-self.max_history:]
        pred = self.predictor.predict_next(tuple(self.history))
        if pred is not None:
            if pred.source_frame_index != frame.frame_index:
                raise ValueError("Predictor source frame mismatch")
            pred.validate_source(updated)
        self.pending_prediction = pred
        self.previous_timestamp_s = frame.timestamp_s
        return FrameResult(frame.frame_index, perception, camera, updated, prior, pred)
