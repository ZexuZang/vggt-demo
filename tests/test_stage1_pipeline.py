"""CPU-only contracts, unrelated to actual deep model accuracy."""
import unittest

import numpy as np

from som_fast4d.core import (Fast4DPipeline, FrameData, MotionPrediction,
                             PerceptionOutput)
from som_fast4d.run_stage1_smoke import (MockMapping, MockPerception,
                                           MockPredictor, MockTracking, synthetic_frame)


class Stage1ContractTests(unittest.TestCase):
    def new_pipeline(self):
        m = MockMapping()
        p = Fast4DPipeline(MockPerception(), MockTracking(), m, MockPredictor())
        return p, m

    def test_observed_mask_semantics(self):
        frame = synthetic_frame(0)
        out = MockPerception().infer(frame)
        out.validate(frame)
        self.assertTrue(out.dynamic_mask[14, 7])
        self.assertFalse(out.static_ba_mask[14, 7])
        self.assertTrue(out.tracking_mask(frame.depth_m)[0, 0])
        out.static_ba_mask[14, 7] = True
        with self.assertRaises(ValueError):
            out.validate(frame)

    def test_frame_validation(self):
        frame = synthetic_frame(0)
        with self.assertRaises(ValueError):
            FrameData(0, 0.0, frame.rgb, frame.depth_m[:5, :], frame.K)

    def test_prediction_consumed_at_next_frame(self):
        pipe, mapping = self.new_pipeline()
        a = pipe.process_frame(synthetic_frame(0))
        self.assertIsNone(a.next_frame_prediction)
        b = pipe.process_frame(synthetic_frame(1))
        self.assertIsNone(b.prior_used_for_current_frame)
        self.assertIsNotNone(b.next_frame_prediction)
        self.assertEqual(b.next_frame_prediction.target_frame_index, 2)
        c = pipe.process_frame(synthetic_frame(2))
        self.assertIsNotNone(c.prior_used_for_current_frame)
        self.assertEqual(c.prior_used_for_current_frame.source_frame_index, 1)
        self.assertEqual(mapping.last_prior_count, 1)
        self.assertAlmostEqual(float(c.gaussian_map.centers_world_m[1, 0]), 1.2, places=4)

    def test_frame_order_rejected(self):
        pipe, _ = self.new_pipeline()
        pipe.process_frame(synthetic_frame(0))
        with self.assertRaises(ValueError):
            pipe.process_frame(synthetic_frame(2))

    def test_wrong_prediction_id_rejected(self):
        pipe, _ = self.new_pipeline()
        pipe.process_frame(synthetic_frame(0))
        b = pipe.process_frame(synthetic_frame(1))
        wrong = MotionPrediction(1, 2, np.array([101], dtype=np.int64),
                                 np.zeros((1, 3), dtype=np.float32),
                                 np.array([True]))
        with self.assertRaises(ValueError):
            wrong.validate_source(b.gaussian_map)


if __name__ == "__main__":
    unittest.main()
