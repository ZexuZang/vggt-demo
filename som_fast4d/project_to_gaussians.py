"""Map sparse scene-flow predictions to dynamic Gaussian centers.

Inputs MUST share a world coordinate frame and metric scale.
This generates an INITIALIZATION PRIOR only: it does not update the Gaussian
optimizer or replace observed scene-flow correction in Flow4DGS-SLAM.
"""
import argparse
from pathlib import Path
import numpy as np


def _nearest(src, tgt, chunk=128):
    try:
        from scipy.spatial import cKDTree
        distance, idx = cKDTree(src).query(tgt, k=1)
        return idx, distance
    except ImportError:
        indices, distances = [], []
        for i in range(0, len(tgt), chunk):
            a = tgt[i:i + chunk]
            dist2 = ((a[:, None, :] - src[None, :, :]) ** 2).sum(axis=-1)
            j = np.argmin(dist2, axis=1)
            indices.append(j)
            distances.append(np.sqrt(dist2[np.arange(len(j)), j]))
        return np.concatenate(indices), np.concatenate(distances)


def propagate(centers, src_points, scene_flow, max_distance=0.2):
    centers = np.asarray(centers, dtype=np.float32)
    src_points = np.asarray(src_points, dtype=np.float32)
    scene_flow = np.asarray(scene_flow, dtype=np.float32)
    for name, x in [("centers", centers), ("src_points", src_points), ("scene_flow", scene_flow)]:
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError("{} must be finite non-empty (N,3), got {}".format(name, x.shape))
    if src_points.shape != scene_flow.shape:
        raise ValueError("source points and scene flow must have the same shape")
    idx, distance = _nearest(src_points, centers)
    valid = distance <= max_distance
    next_centers = centers.copy()
    next_centers[valid] += scene_flow[idx[valid]]
    return next_centers, valid, distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gaussian-centers", required=True, help=".npy shape [M,3], dynamic only, world frame")
    p.add_argument("--prediction", required=True, help="GenFlow3D .npz from genflow3d_predict.py")
    p.add_argument("--output", required=True, help="Output .npz initialization priors")
    p.add_argument("--max-distance", type=float, default=0.2, help="Nearest-point limit, meters")
    a = p.parse_args()
    centers = np.load(a.gaussian_centers, allow_pickle=False)
    with np.load(a.prediction, allow_pickle=False) as pred:
        if str(pred["coordinate_frame"]) != "world":
            raise ValueError("Prediction coordinates must be in metric world frame")
        new_centers, valid, dist = propagate(centers, pred["source_points"], pred["future_flow"], a.max_distance)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, previous_centers=centers, predicted_centers=new_centers,
                        has_prior=valid, nearest_distance=dist)
    print("Saved {} predicted dynamic Gaussian centers ({} matched / {}).".format(
        out, int(valid.sum()), len(valid)))
    print("Only an initialization prior. New-frame depth/flow should correct this in the SLAM optimizer.")


if __name__ == "__main__":
    main()
