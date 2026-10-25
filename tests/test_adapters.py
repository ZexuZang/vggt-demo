import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from som_fast4d.prepare_masks import run
from som_fast4d.project_to_gaussians import propagate
from som_fast4d.genflow3d_predict import select_points


class IntegrationAdaptersTest(unittest.TestCase):
    def test_preours_mask_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "seq"
            pre = root / "pre"
            for i in range(2):
                (ds / "color").mkdir(parents=True, exist_ok=True)
                (pre / "{:06d}".format(i)).mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6), color=(20, 30, 40)).save(ds / "color" / "{:06d}.png".format(i))
                mask = np.zeros((6, 8), dtype=np.uint8)
                mask[1:4, 2:5] = 255
                Image.fromarray(mask).save(pre / "{:06d}".format(i) / "refined_dynamic_mask.png")
            out = ds / "mask_colour"
            manifest = run(ds, pre, out)
            self.assertEqual(len(manifest), 2)
            self.assertEqual(int(np.asarray(Image.open(out / "000001.png"))[2, 3]), 255)
            self.assertEqual(int(np.asarray(Image.open(out / "000001.png"))[0, 0]), 0)
            with self.assertRaises(FileExistsError):
                run(ds, pre, out)

    def test_scene_flow_nearest(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        sf = np.array([[0, 0.1, 0], [0, 0.2, 0]], dtype=np.float32)
        centers = np.array([[0.02, 0, 0], [1, 0, 0], [4, 0, 0]], dtype=np.float32)
        pred, good, dist = propagate(centers, src, sf, max_distance=0.1)
        self.assertTrue(np.array_equal(good, np.array([True, True, False])))
        self.assertAlmostEqual(float(pred[0, 1]), 0.1, places=5)
        self.assertAlmostEqual(float(pred[1, 1]), 0.2, places=5)
        self.assertAlmostEqual(float(pred[2, 1]), 0.0, places=5)

    def test_point_sampling(self):
        src = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.assertEqual(select_points(src, 16, seed=1).shape, (16, 3))
        self.assertTrue(np.array_equal(select_points(src, 16, 1), select_points(src, 16, 1)))


if __name__ == "__main__":
    unittest.main()
