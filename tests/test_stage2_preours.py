"""Tests for the saved-output Pre-Ours adapter, no GPU or actual models."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from som_fast4d.core import FrameData
from som_fast4d.perception import PreOursPerceptionAdapter
from som_fast4d.make_stage2_clip import make_clip
from som_fast4d.run_stage2_preours import run


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ds=self.root/"seq"
        self.pre=self.root/"pre"
        for d in (self.ds/"color",self.ds/"depth"):
            d.mkdir(parents=True,exist_ok=True)
        self.imgs=[]
        for i in range(3):
            path=self.ds/"color"/("%06d.png"%i)
            Image.new("RGB",(8,6),color=(30,60,90)).save(path)
            Image.fromarray(np.full((6,8),10000,dtype=np.uint16)).save(self.ds/"depth"/path.name)
            self.imgs.append(path)
            od=self.pre/path.stem
            od.mkdir(parents=True)
            dyn=np.zeros((6,8),dtype=np.uint8)
            dyn[1:4,2:5]=255
            Image.fromarray(dyn).save(od/"refined_dynamic_mask.png")
            conf=np.full((6,8),0.9,dtype=np.float32)
            conf[1:4,2:5]=0.02
            np.save(od/"static_weight.npy",conf)
            Image.fromarray((conf>0.5).astype(np.uint8)*255).save(od/"static_ba_mask.png")
            (od/"summary.json").write_text(json.dumps({"image_path":str(path)}))

    def frame(self,i=0):
        return FrameData(i,i/30.0,np.asarray(Image.open(self.imgs[i]).convert("RGB")),
                         np.full((6,8),2.,dtype=np.float32),np.array([[10.,0,4],[0,10,3],[0,0,1]]))

    def adapter(self,**kwargs):
        return PreOursPerceptionAdapter(str(self.pre),[str(p) for p in self.imgs],**kwargs)

    def test_real_file_mask_semantics(self):
        f=self.frame()
        output=self.adapter().infer(f)
        self.assertTrue(output.dynamic_mask[2,3]);self.assertFalse(output.static_ba_mask[2,3])
        self.assertTrue(output.static_ba_mask[0,0]);self.assertLess(output.static_confidence[2,3],0.1)
        self.assertEqual(output.tracking_mask(f.depth_m).shape,(6,8))

    def test_edge_preferred_and_overlap_rejected(self):
        od=self.pre/"000000"
        Image.fromarray(np.full((6,8),255,dtype=np.uint8)).save(od/"edge_refined_dynamic_mask.png")
        np.save(od/"edge_refined_static_weight.npy",np.ones((6,8),dtype=np.float32))
        np.save(od/"edge_refined_static_ba_mask.npy",np.ones((6,8),dtype=np.uint8))
        out=self.adapter().infer(self.frame())
        self.assertTrue(out.dynamic_mask.all())
        self.assertFalse(out.static_ba_mask.any())

    def test_alignment_mismatch(self):
        od=self.pre/"000000"/"summary.json"
        od.write_text(json.dumps({"image_path":"bad_frame.png"}))
        with self.assertRaisesRegex(ValueError,"alignment"):
            self.adapter().infer(self.frame())

    def test_missing_confidence_fails(self):
        (self.pre/"000000"/"static_weight.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            self.adapter().infer(self.frame())

    def test_resize_requires_explicit_flag(self):
        od=self.pre/"000000"
        Image.fromarray(np.full((3,4),255,dtype=np.uint8)).save(od/"refined_dynamic_mask.png")
        with self.assertRaisesRegex(ValueError,"resolution"):
            self.adapter().infer(self.frame())
        out=self.adapter(resize_masks=True).infer(self.frame())
        self.assertEqual(out.dynamic_mask.shape,(6,8))
        self.assertTrue(out.dynamic_mask.all())

    def test_stage2_pipeline_mock(self):
        results=run(str(self.ds),str(self.pre),str(self.root/"report"),
                    10.,10.,4.,3.,5000.,30.,3)
        self.assertEqual(len(results),3)
        self.assertGreater(results[0]["static_ba_fraction"],0.4)
        self.assertTrue((self.root/"report"/"000001_overlay.png").is_file())
        rep=json.loads((self.root/"report"/"report.json").read_text())
        self.assertTrue(rep["not_realtime"])

    def test_clip_makes_aligned_frames(self):
        clip=self.root/"short_clip"
        rows=make_clip(self.ds,clip,0,2)
        self.assertEqual(len(rows),2)
        self.assertTrue((clip/"color"/"000001.png").is_file())
        self.assertTrue((clip/"depth"/"000001.png").is_file())
        with self.assertRaises(FileExistsError):make_clip(self.ds,clip,0,2)


if __name__ == "__main__":
    unittest.main()
