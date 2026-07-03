# VGGT + Any4D + CoTracker Motion-Aware DROID-W

## Overview

## Method Pipeline

## Repository Structure

## Colab Usage

### 0. Environment Setup
Run notebooks/00_environment_droidw_colab.ipynb

### 1. Generate Ours Static Weights
Run notebooks/01_generate_ours_static_weight.ipynb

### 2. Run DROID-W Baseline and Ours
Run notebooks/02_run_droidw_baseline_vs_ours.ipynb

### 3. Evaluate
Run notebooks/03_evaluate_results.ipynb

## Metrics

- ATE RMSE
- ATE Mean
- ATE Median
- ATE Max
- RPE translation
- RPE rotation
- Runtime FPS
- Static weight statistics

## Notes on DROID-W Environment
DROID-W requires several CUDA extensions and source patches on Colab.
The setup is automated in scripts/env/setup_droidw_colab.sh.