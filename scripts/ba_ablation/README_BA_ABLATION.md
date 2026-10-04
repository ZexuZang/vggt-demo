# VGGT + Any4D + CoTracker + DROID-W: BA Ablation Scripts

These scripts are written against the current structure of `ZexuZang/vggt-demo`.

## Variants

- **Baseline 0**: original DROID-W, no external dynamic/static BA weighting.
- **Baseline 1**: soft isotropic refined-mask BA weighting, `w = 1 - P_dyn`.
- **Ours**: same dense refined reliability plus camera-compensated motion-direction anisotropic BA weighting (diagonal Mahalanobis approximation).

The refined dense dynamic probability is the source of `P_dyn`; it is **not** rebuilt from sparse CoTracker rasterization. CoTracker is used only to estimate the residual motion direction for Ours.

## Recommended folder in the repository

Copy these files into:

```text
vggt-demo/scripts/ba_ablation/
```

## 1. Generate Baseline 1 and Ours weights

Example for Bonn `person_tracking`:

```bash
cd /content/vggt-demo

python scripts/ba_ablation/01_generate_ba_ablation_weights.py \
  --tracks_npz outputs/ours_bonn_person_temporal8/cotracker_any4d_tracks.npz \
  --image_dir /content/drive/MyDrive/Datasets/Bonn/person_tracking/rgb \
  --refined_dynamic_path /path/to/refined_dynamic_probability_per_frame \
  --output_dir outputs/ba_ablation_weights/person_tracking \
  --min_weight 0.05 \
  --static_seed_thresh 0.25 \
  --kappa_max 12.0
```

Outputs:

```text
outputs/ba_ablation_weights/person_tracking/
├── baseline1/
│   └── static_weight_npy/
├── ours/
│   └── precision_weight_npy/
├── diagnostics/
└── summary.json
```

## 2. Patch DROID-W once for two-channel Ours weights

```bash
python scripts/ba_ablation/02_patch_droidw_for_anisotropic_weight.py \
  --repo_root /content/vggt-demo
```

Scalar Baseline-1 files remain compatible because they are automatically repeated to x/y.

## 3. Run all three variants

```bash
python scripts/ba_ablation/03_run_baseline0_baseline1_ours.py \
  --repo_root /content/vggt-demo \
  --weights_root /content/vggt-demo/outputs/ba_ablation_weights/person_tracking \
  --output_root /content/vggt-demo/outputs_compare/ba_ablation
```

This creates:

```text
outputs_compare/ba_ablation/
├── baseline0/
├── baseline1/
└── ours/
```

To test only one method:

```bash
python scripts/ba_ablation/03_run_baseline0_baseline1_ours.py \
  --repo_root /content/vggt-demo \
  --weights_root /content/vggt-demo/outputs/ba_ablation_weights/person_tracking \
  --output_root /content/vggt-demo/outputs_compare/ba_ablation \
  --variants baseline1
```

## 4. Evaluate three-way ATE/RPE

```bash
python scripts/ba_ablation/04_evaluate_baseline0_baseline1_ours.py \
  --repo_root /content/vggt-demo \
  --root_dir /content/vggt-demo/outputs_compare/ba_ablation
```

Main table:

```text
outputs_compare/ba_ablation/eval_three_way/ba_ablation_metrics.csv
```

The evaluator reuses your existing `scripts/eval/droidw_compare_eval.py` implementation so the ATE/RPE calculation remains consistent with your current repository.
