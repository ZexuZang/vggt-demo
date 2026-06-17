# Push this patch to GitHub

From your local repo root:

```bash
git checkout -b any4d-dynamic-mask
# copy scripts/, notebooks/, docs/ from this patch into your repo
git add scripts/any4d_mask notebooks/VGGT_Any4D_Dynamic_Mask_Colab.ipynb docs/VGGT_ANY4D_DYNAMIC_MASK.md
git commit -m "Add Any4D dynamic mask refinement for VGGT"
git push origin any4d-dynamic-mask
```

Then open a pull request on GitHub, or merge this branch into main.
