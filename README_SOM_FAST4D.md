# SOM Fast4D — VS Code Starter Adapters (NOT YET a finished SLAM integration)

Source checked: ZexuZang/vggt-demo branch `experiments/som-fast4d-v1`,
Flow4DGS-SLAM `main`, GenFlow3D `main` as accessed 2026-10-25.

## Files

- `som_fast4d/prepare_masks.py`: converts **already computed** per-frame Pre-Ours dynamic masks into original Flow4DGS-SLAM `mask_colour/` format. Does not execute VGGT/Any4D.
- `som_fast4d/patch_flow_dataset.py`: minimal numeric sort / count sanity patch for Flow4DGS-SLAM `utils/dataset.py`.
- `som_fast4d/genflow3d_predict.py`: invokes OFFICIAL GenFlow3D model to predict one unobserved future scene flow (requires pretrained checkpoint, torch, CUDA, PointNet2 ops).
- `som_fast4d/project_to_gaussians.py`: nearest-neighbor projected initialization prior from predicted flow to supplied dynamic Gaussian centers. **Not yet wired into Flow4DGS-SLAM optimizer.**
- `tests/test_adapters.py`: CPU smoke tests for interfaces. Cannot validate CUDA inference or final SLAM.

## Clone and prepare

```bash
git clone --branch experiments/som-fast4d-v1 https://github.com/ZexuZang/vggt-demo.git
cd vggt-demo
git switch -c experiments/som-fast4d-integration-v1
# Unzip this package directly in the vggt-demo root (it adds som_fast4d and tests).
python -m pip install numpy pillow
python -m unittest discover -s tests -v
```

Keep independent checkouts of Flow4DGS-SLAM and GenFlow3D:

```bash
cd third_party
git clone https://github.com/wangys16/Flow4DGS-SLAM.git
git clone https://github.com/ustc-hlli/GenFlow3D.git
cd ..
```

For Flow4DGS environment use its official README: Python 3.8, compatible CUDA 11.7/Torch 1.13.1; for GenFlow3D use a **separate** environment with its official PointNet2 extension. Adapt versions to your GPU architecture. Do NOT blindly run both models within one mixed environment.

## Stage 1 — Pre-Ours -> Flow4DGS masks

First, run existing Pre-Ours on your video to save a per-frame directory:

```
outputs/preours_per_frame/
    000000/refined_dynamic_mask.png
    000001/refined_dynamic_mask.png
    000002/refined_dynamic_mask.png
    ...
```

Each mask MUST correspond to the same frame in
`third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking/color/`.
A single Pre-Ours `ref_idx=0` output does NOT cover all video frames. Audit any
future-frame use in the original Any4D/VGGT inference before claiming online.

```bash
python -m som_fast4d.prepare_masks \
  --dataset third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking \
  --preours outputs/preours_per_frame \
  --output third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking/mask_colour
python -m som_fast4d.patch_flow_dataset --file third_party/Flow4DGS-SLAM/utils/dataset.py
```

Stage 1 run (inside Flow4DGS-SLAM's own environment, from its root):

```bash
python slam.py \
  --inherit_from configs/rgbd/bonn/config.yaml \
  --config configs/rgbd/bonn/bonn_ps_trk.yaml \
  --eval --dynamic --yolo 0 --exp_name preours_mask
```

The dataset returns `~mask` = static True, and `flow_mask` can still refine this
further while `flow_cam` is retained for ego-motion initialization.
For mask ablations keep the rest of the configuration identical.

## Stage 2 — GenFlow3D future prediction

Export **camera-motion-compensated, world-frame, metric** sequential point
clouds to `outputs/world_points/frame_000000.npy` etc. Each file is `[Ni,3]`.
Until you add a point-cloud exporter, do not use raw camera-coordinate depth
clouds directly: their apparent displacement mixes camera and object motion.

Official GenFlow3D main model from `models.py` is `GenFlow_rec_seq`. It expects
`pcs` / `feats` as lists of `[B,N,3]`; the official nuScenes pretrain config is
`npoint=8192`, `length_pre=3`, `length_fut=1`.

Run in separate GenFlow3D environment:

```bash
python -m som_fast4d.genflow3d_predict \
  --history-dir outputs/world_points \
  --model-root third_party/GenFlow3D \
  --checkpoint third_party/GenFlow3D/pretrain/GenFlow3d_nuscenes.pth \
  --history 3 --npoint 8192 --horizon 1 \
  --output outputs/prediction/future_flow.npz
```

Caution: the official checkpoint is trained on automotive point clouds; indoor
RGB-D motion needs domain validation or fine-tuning. The script does NOT run
GenFlow3D during live Flow4DGS mapping yet.

## Stage 3 — Gaussian prediction initializer

Export the current DYNAMIC Gaussian centers to `.npy` (shape `[M,3]`) in the SAME
world-coordinate frame. The supplied script only computes a suggested next-frame
center: `mu_next = mu + predicted_flow_nn`.

```bash
python -m som_fast4d.project_to_gaussians \
  --gaussian-centers outputs/dynamic_centers_t.npy \
  --prediction outputs/prediction/future_flow.npz \
  --output outputs/prediction/gaussian_init_t1.npz
```

The critical remaining CUDA integration is to inject these predicted centers
as **INITIAL VALUES** into Flow4DGS-SLAM's mapping, then correct them with
real target-frame RGB-D/optical-flow observations. It requires careful coupling
to the Gaussian model's representation, optimizer state, and multiprocessing;
this starter deliberately does not overwrite those internals.

## Explicit limitations

1. Original Pre-Ours sliding windows could see future frames: no realtime claim yet.
2. Predicted Flow is **not** observed flow and cannot replace measurement update.
3. GenFlow3D pretraining domain differs (nuScenes/AV2 vs Bonn/TUM).
4. Mapping / camera BA changes and full end-to-end FPS have NOT been tested.
5. The supplied CPU smoke tests check contracts, not tracking/4D rendering accuracy.

## Optional: generate offline per-frame Pre-Ours masks (pilot only)

Original `refine_dynamic_mask.py` produces a mask for ONE reference frame, so
running it once is not enough for the whole Bonn dataset. This helper drives
existing scripts for each frame, but does not bypass model dependencies.

First export VGGT predictions in its own environment on the `color/` sequence:

```bash
python scripts/any4d_mask/export_vggt_predictions.py \
  --image_dir third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking/color \
  --output_npz outputs/vggt_all.npz
```

Then from the `vggt-demo` root, with configured environment-specific interpreters:

```bash
python -m som_fast4d.run_preours_perframe \
  --image-dir third_party/Flow4DGS-SLAM/datasets/bonn/rgbd_bonn_person_tracking/color \
  --vggt-npz outputs/vggt_all.npz \
  --any4d-root third_party/Any4D \
  --any4d-checkpoint third_party/Any4D/checkpoints/any4d_4v_combined.pth \
  --any4d-python /PATH/TO/any4d/env/bin/python \
  --refine-python /PATH/TO/vggt/env/bin/python \
  --out outputs/preours_per_frame \
  --start-index 0 --end-index 5 --mode next
```

**`--mode next` uses frames from the future and is NOT real-time.** It is for
validating a controlled offline mask swap. Use `--mode past --start-index 1`
for a causal motion target once frame-0 initialization is separately solved.
Running the original VGGT exporter on an entire sequence may exceed GPU RAM;
for pilot tests use a small complete clip with matching frame indices.
