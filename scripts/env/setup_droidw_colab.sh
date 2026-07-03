#!/usr/bin/env bash
set -e

REPO_ROOT=${REPO_ROOT:-/content/vggt-demo}
DROIDW_ROOT="$REPO_ROOT/third_party/DROID-W"

echo "==== DROID-W Colab environment setup ===="
echo "REPO_ROOT=$REPO_ROOT"
echo "DROIDW_ROOT=$DROIDW_ROOT"

cd "$REPO_ROOT"

echo "==== 1. Update submodules / clone DROID-W ===="
git submodule update --init --recursive || true
if [ ! -d "$DROIDW_ROOT" ]; then
  git clone https://github.com/MoyangLi00/DROID-W.git "$DROIDW_ROOT"
fi

cd "$DROIDW_ROOT"

echo "==== Update DROID-W submodules ===="
git submodule sync --recursive
git submodule update --init --recursive

echo "==== 2. Install Python dependencies ===="
python -m pip install -q \
  ninja pybind11 munch pyyaml tqdm scipy opencv-python imageio imageio-ffmpeg \
  matplotlib tensorboardX einops colorama natsort yacs gdown open3d evo \
  glfw PyOpenGL PyOpenGL-accelerate imgui imgviz PyGLM \
  mmengine mmcv-lite timm addict yapf

echo "==== 3. Install torch_scatter ===="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
PY

python -m pip install -q torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html || true

echo "==== 4. Apply source patches ===="
python "$REPO_ROOT/scripts/env/patch_droidw_colab.py"

echo "==== 5. Build droid_backends ===="
cd "$DROIDW_ROOT"
rm -rf build *.egg-info
find . -name "*.so" -delete || true

export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="7.5"

python -m pip install -e . --no-build-isolation -v

echo "==== 6. Build lietorch ===="
cd "$DROIDW_ROOT"
rm -rf thirdparty/lietorch/build thirdparty/lietorch/*.egg-info
find thirdparty/lietorch -name "*.so" -delete || true
python -m pip install -e thirdparty/lietorch --no-build-isolation -v

echo "==== 7. Build diff gaussian rasterization ===="
cd "$DROIDW_ROOT"

if [ -d thirdparty/diff-gaussian-rasterization-w-pose ]; then
  python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose --no-build-isolation -v
elif [ -d thirdparty/diff-gaussian-rasterization ]; then
  python -m pip install -e thirdparty/diff-gaussian-rasterization --no-build-isolation -v
elif [ -d thirdparty/gaussian_splatting/submodules/diff-gaussian-rasterization ]; then
  python -m pip install -e thirdparty/gaussian_splatting/submodules/diff-gaussian-rasterization --no-build-isolation -v
else
  echo "WARNING: diff-gaussian-rasterization folder not found."
fi

echo "==== 8. Build simple-knn ===="
cd "$DROIDW_ROOT"

if [ -d thirdparty/simple-knn ]; then
  python -m pip install -e thirdparty/simple-knn --no-build-isolation -v
elif [ -d thirdparty/gaussian_splatting/submodules/simple-knn ]; then
  python -m pip install -e thirdparty/gaussian_splatting/submodules/simple-knn --no-build-isolation -v
else
  echo "WARNING: simple-knn folder not found."
fi

echo "==== 9. Download DROID pretrained model ===="
cd "$DROIDW_ROOT"
mkdir -p pretrained

if [ ! -f pretrained/droid.pth ]; then
  wget -O pretrained/droid.pth https://huggingface.co/vslamlab/droidslam/resolve/main/droid.pth
fi

echo "==== 10. Validate imports ===="
bash "$REPO_ROOT/scripts/env/check_droidw_env.sh"

echo "==== DROID-W Colab environment setup finished ===="