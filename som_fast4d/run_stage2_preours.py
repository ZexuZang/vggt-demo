"""Stage 2 integration verification: REAL saved Pre-Ours + MOCK other components.

This runner loads real RGB-D and Pre-Ours outputs via the Stage-1 Pipeline.
It does NOT run Pre-Ours inference and does NOT run Flow4DGS / GenFlow3D.
Any future-frame VGGT/Any4D context makes this OFFLINE NON-CAUSAL validation.

python -m som_fast4d.run_stage2_preours --dataset data/stage2_bonn_6 \
  --preours-dir outputs/stage2/preours --fx 542.82 --fy 542.58 \
  --cx 315.59 --cy 237.76 --depth-scale 5000 --resize-masks
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import Fast4DPipeline, FrameData
from .perception import PreOursPerceptionAdapter
from .run_stage1_smoke import MockTracking, MockMapping, MockPredictor


def _collect(dataset: Path, max_frames: int):
    color_dir, depth_dir = dataset / "color", dataset / "depth"
    rgb_paths=sorted([x for x in color_dir.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg")])
    depth_by_stem={x.stem:x for x in depth_dir.iterdir() if x.suffix.lower() in (".png", ".tiff", ".tif")}
    rgb_paths=rgb_paths[:max_frames]
    if not rgb_paths:
        raise FileNotFoundError("No RGB frames in " + str(color_dir))
    missing=[x.stem for x in rgb_paths if x.stem not in depth_by_stem]
    if missing:
        raise FileNotFoundError("Missing matching depth for RGB stems: " + str(missing))
    return rgb_paths, depth_by_stem


def load_frame(index: int, image_path: Path, depth_path: Path, K: np.ndarray, depth_scale: float, fps: float) -> FrameData:
    rgb=np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    with Image.open(depth_path) as dp:
        depth=np.asarray(dp, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth must be single-channel: " + str(depth_path))
    if depth_scale <= 0 or fps <= 0:
        raise ValueError("Depth scale and frame rate must be positive")
    return FrameData(index, index/fps, rgb, depth/depth_scale, K.copy())


def overlay(rgb: np.ndarray, dynamic: np.ndarray, static_ba: np.ndarray) -> Image.Image:
    img=rgb.copy().astype(np.float32)
    img[dynamic]=img[dynamic]*0.55 + np.array([255,64,64])*0.45
    img[static_ba]=img[static_ba]*0.78+np.array([30,210,100])*0.22
    return Image.fromarray(img.clip(0,255).astype(np.uint8))


def run(dataset: str, preours: str, out: str, fx: float, fy: float, cx: float, cy: float,
        depth_scale: float = 5000., fps: float = 30., max_frames: int = 6,
        resize_masks: bool = False, static_weight_threshold: float = 0.45,
        min_static_fraction: float = 0.01):
    if max_frames < 2:
        raise ValueError("Need at least two frames")
    dataset,out=Path(dataset),Path(out)
    rgb_paths, depths=_collect(dataset,max_frames)
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],dtype=np.float64)
    perception=PreOursPerceptionAdapter(preours,[str(p) for p in rgb_paths],
                resize_masks=resize_masks,static_weight_threshold=static_weight_threshold)
    mapping=MockMapping()
    pipeline=Fast4DPipeline(perception,MockTracking(),mapping,MockPredictor())
    # no output files are written until frame loading and mask validation succeeds
    frames=[]; masks=[]
    for i,im in enumerate(rgb_paths):
        frame=load_frame(i,im,depths[im.stem],K,depth_scale,fps)
        percept=perception.infer(frame)
        pstat=float(percept.tracking_mask(frame.depth_m).mean())
        if pstat < min_static_fraction:
            raise ValueError("Not enough valid static pixels in %s: %.4f" %(im.name,pstat))
        frames.append(frame)
        masks.append(percept)
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    print("Stage 2: REAL Pre-Ours masks + MOCK tracking/mapping/prediction")
    for frame in frames:
        r=pipeline.process_frame(frame)
        mask=r.perception
        rgb_path=rgb_paths[frame.frame_index]
        image_out=out/(rgb_path.stem+"_overlay.png")
        overlay(frame.rgb,mask.dynamic_mask,mask.static_ba_mask).save(image_out)
        vals={"frame":rgb_path.name,"dynamic_fraction":float(mask.dynamic_mask.mean()),
              "static_ba_fraction":float(mask.static_ba_mask.mean()),
              "valid_tracking_fraction":float(mask.tracking_mask(frame.depth_m).mean()),
              "confidence_mean":float(mask.static_confidence.mean()),
              "prior_used": int(mapping.last_prior_count),"overlay":str(image_out)}
        report.append(vals)
        print("frame=%s dynamic=%.3f static_ba=%.3f tracking=%.3f"%(
            rgb_path.name,vals["dynamic_fraction"],vals["static_ba_fraction"],vals["valid_tracking_fraction"]))
    (out/"report.json").write_text(json.dumps({
       "validation_type":"offline precomputed real Pre-Ours outputs; other components are mock",
       "not_realtime":True,"preours_dir":str(Path(preours).resolve()),
       "dataset":str(dataset.resolve()),"frames":report},indent=2),encoding="utf8")
    print("PASS: Pre-Ours real outputs match Stage-1 PerceptionOutput, %d frames; see %s" %(len(report),out))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True)
    p.add_argument("--preours-dir",required=True)
    p.add_argument("--out",default="outputs/stage2/validation")
    for k in ("fx","fy","cx","cy"):
        p.add_argument("--"+k,type=float,required=True)
    p.add_argument("--depth-scale",type=float,default=5000.)
    p.add_argument("--fps",type=float,default=30.)
    p.add_argument("--max-frames",type=int,default=6)
    p.add_argument("--resize-masks",action="store_true")
    p.add_argument("--static-weight-threshold",type=float,default=0.45)
    p.add_argument("--min-static-fraction",type=float,default=0.01)
    a=p.parse_args()
    run(a.dataset,a.preours_dir,a.out,a.fx,a.fy,a.cx,a.cy,a.depth_scale,a.fps,a.max_frames,
        a.resize_masks,a.static_weight_threshold,a.min_static_fraction)


if __name__=="__main__":
    main()
