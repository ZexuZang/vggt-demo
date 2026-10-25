"""Adapt official GenFlow3D to future scene-flow prediction on .npy point clouds.

Point clouds must be finite [Ni,3] world-frame metric coordinates, aligned
across all history frames by CAMERA EGO-MOTION COMPENSATION. This does not
train on indoor RGB-D data, and pretrained automotive weights may not
transfer. Requires the official GenFlow3D code, PointNet2 CUDA ops, torch,
and an official compatible checkpoint. All inference is done in a dedicated
environment (do not install over Flow4DGS-SLAM's torch environment).
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np


def _natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def select_points(points, npoint, seed):
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Point cloud must be [N,3]")
    valid = np.isfinite(arr).all(axis=1)
    arr = arr[valid]
    if len(arr) == 0:
        raise ValueError("Point cloud has no finite points")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(arr), size=npoint, replace=len(arr) < npoint)
    return np.ascontiguousarray(arr[ids])


def load_model(root, checkpoint, npoint, device):
    import torch
    root = str(Path(root).resolve())
    if not Path(root, "models.py").is_file():
        raise FileNotFoundError("Invalid GenFlow3D root (models.py absent): " + root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import models  # official GenFlow3D implementation
    model = models.GenFlow_rec_seq(npoint).to(device).eval()
    state = torch.load(checkpoint, map_location=device)  # load only trusted checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def predict(root, checkpoint, history_points, npoint=8192, horizon=1, seed=0, device="cuda"):
    import torch
    if len(history_points) < 2:
        raise ValueError("Need at least 2 observed historical point clouds")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official PointNet2 ops; use an NVIDIA GPU environment")
    sampled = [select_points(pc, npoint, seed+i) for i,pc in enumerate(history_points)]
    pcs = [torch.from_numpy(pc).unsqueeze(0).to(device) for pc in sampled]  # [B,N,3]
    feats = [pc.clone() for pc in pcs]  # official nuScenes dataset uses xyz as features
    # In official models.py, during eval() the gts list is replaced by None.
    # Dummy tensors exist only to specify time steps and batch size.
    empty_flow = [torch.zeros_like(pcs[-1]) for _ in range(len(pcs)-1+horizon)]
    model = load_model(root, checkpoint, npoint, device)
    with torch.no_grad():
        pred_flows, _, _ = model(pcs, feats, empty_flow)
    first_future_index = len(pcs) - 1
    flow = pred_flows[0][first_future_index]
    if flow.ndim != 3 or flow.shape[0] != 1:
        raise RuntimeError("Unexpected GenFlow3D output flow shape: " + str(flow.shape))
    if flow.shape[1] == 3 and flow.shape[2] == npoint:
        flow = flow[0].transpose(0,1)
    elif flow.shape[1] == npoint and flow.shape[2] == 3:
        flow = flow[0]
    else:
        raise RuntimeError("Expected GenFlow3D finest flow [1,3,N] or [1,N,3], got " + str(flow.shape))
    future_flow = flow.detach().float().cpu().numpy()
    return sampled[-1], future_flow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir", required=True, help="Directory with frame-sorted .npy [N,3] world-frame clouds")
    p.add_argument("--model-root", required=True, help="Official GenFlow3D checkout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Output .npz")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--npoint", type=int, default=8192)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    paths = sorted(Path(a.history_dir).glob("*.npy"), key=_natural_key)
    if len(paths) < a.history:
        raise ValueError("Only {} point cloud files; history={} requested".format(len(paths), a.history))
    paths = paths[-a.history:]
    points = [np.load(path, allow_pickle=False) for path in paths]
    source, flow = predict(a.model_root, a.checkpoint, points,
                           a.npoint, a.horizon, a.seed, a.device)
    if not np.isfinite(flow).all():
        raise RuntimeError("GenFlow3D produced non-finite flow")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source_points=source, future_flow=flow,
                        predicted_points=source+flow,
                        coordinate_frame=np.array("world"),
                        input_history=np.array([p.name for p in paths]))
    print("Saved {} with scene flow shape {}".format(out, flow.shape))
    print("WARNING: automotive checkpoint on indoor Gaussian data needs evaluation/fine-tuning")
    print("WARNING: predicted movement requires correction using next observed RGB-D frame")


if __name__ == "__main__":
    main()
