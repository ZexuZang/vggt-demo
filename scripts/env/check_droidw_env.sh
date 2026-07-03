#!/usr/bin/env bash
set -e

REPO_ROOT=${REPO_ROOT:-/content/vggt-demo}
DROIDW_ROOT="$REPO_ROOT/third_party/DROID-W"

cd "$DROIDW_ROOT"

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export PYTHONPATH="$DROIDW_ROOT:$PYTHONPATH"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)

import munch
import colorama
import imgviz
import glm
import glfw
import OpenGL
import imgui

from torch_scatter import scatter_mean

import droid_backends
import lietorch

try:
    import diff_gaussian_rasterization
    print("diff_gaussian_rasterization OK")
except Exception as e:
    print("diff_gaussian_rasterization FAILED:", e)
    raise

try:
    from simple_knn._C import distCUDA2
    print("simple_knn OK")
except Exception as e:
    print("simple_knn FAILED:", e)
    raise

print("DROID-W environment check OK")
PY