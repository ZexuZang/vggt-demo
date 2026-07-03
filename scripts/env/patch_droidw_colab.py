#!/usr/bin/env python3
from pathlib import Path
import py_compile

REPO_ROOT = Path("/content/vggt-demo")
DROIDW_ROOT = REPO_ROOT / "third_party" / "DROID-W"

def replace_in_file(path: Path, replacements):
    if not path.exists():
        print(f"[SKIP] Missing {path}")
        return
    text = path.read_text()
    old = text
    for a, b in replacements:
        text = text.replace(a, b)
    if text != old:
        path.write_text(text)
        print(f"[PATCHED] {path}")
    else:
        print(f"[OK] {path}")

def patch_droid_backends():
    replace_in_file(
        DROIDW_ROOT / "src/lib/altcorr_kernel.cu",
        [
            ("fmap1.type()", "fmap1.scalar_type()"),
            ("fmap2.type()", "fmap2.scalar_type()"),
        ],
    )
    replace_in_file(
        DROIDW_ROOT / "src/lib/correlation_kernels.cu",
        [
            ("volume.type()", "volume.scalar_type()"),
        ],
    )

def patch_lietorch():
    root = DROIDW_ROOT / "thirdparty/lietorch/lietorch"
    if not root.exists():
        print("[SKIP] lietorch source not found")
        return

    files = []
    for ext in ["*.cpp", "*.cu", "*.h", "*.hpp"]:
        files.extend(root.rglob(ext))

    for p in files:
        text = p.read_text()
        old = text

        # Tensor dtype dispatch
        text = text.replace(".type()", ".scalar_type()")

        # But device check must stay device().type()
        text = text.replace(".device().scalar_type()", ".device().type()")

        # CUDA check should use tensor.is_cuda()
        text = text.replace(".scalar_type().is_cuda()", ".is_cuda()")

        if text != old:
            p.write_text(text)
            print(f"[PATCHED] {p}")

def patch_simple_knn():
    p = DROIDW_ROOT / "thirdparty/simple-knn/simple_knn.cu"
    if not p.exists():
        print("[SKIP] simple_knn.cu not found")
        return

    text = p.read_text()
    if "#include <cfloat>" not in text and "#include <float.h>" not in text:
        lines = text.splitlines()
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#include"):
                insert_idx = i + 1
        lines.insert(insert_idx, "#include <cfloat>")
        p.write_text("\n".join(lines) + "\n")
        print(f"[PATCHED] Added <cfloat> to {p}")
    else:
        print(f"[OK] {p}")

def patch_numpy_2():
    replace_in_file(
        DROIDW_ROOT / "src/utils/datasets.py",
        [
            ("np.unicode_", "np.str_"),
        ],
    )

def patch_fit3d_feature_extractor():
    p = DROIDW_ROOT / "src/utils/mono_priors/img_feature_extractors.py"
    if not p.exists():
        print("[SKIP] img_feature_extractors.py not found")
        return

    text = p.read_text()

    if "def get_intermediate_layers(self, x, n=1, reshape=None, **kwargs):" in text:
        print("[OK] Fit3D get_intermediate_layers already patched")
        py_compile.compile(str(p), doraise=True)
        return

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("    def get_intermediate_layers"):
            start = i
            break

    if start is None:
        print("[SKIP] Cannot find get_intermediate_layers")
        return

    end = None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("    def "):
            end = j
            break
        if lines[j].startswith("def predict_img_features"):
            end = j
            break

    if end is None:
        print("[SKIP] Cannot determine end of get_intermediate_layers")
        return

    new_func = r'''    def get_intermediate_layers(self, x, n=1, reshape=None, **kwargs):
        outputs = None

        if hasattr(self.model, "get_intermediate_layers"):
            try:
                outputs = self.model.get_intermediate_layers(x, n=n, reshape=True)
            except TypeError:
                try:
                    outputs = self.model.get_intermediate_layers(x, n=n)
                except TypeError:
                    outputs = self.model.get_intermediate_layers(x, n)

        if outputs is None and hasattr(self.model, "_intermediate_layers"):
            outputs = self.model._intermediate_layers(x, n)

        if outputs is None:
            raise RuntimeError("No compatible intermediate layer API found.")

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        features = []
        for out in outputs:
            if isinstance(out, (list, tuple)):
                out = out[0]

            if out.ndim == 4:
                if hasattr(self.model, "norm"):
                    try:
                        out = self.model.norm(
                            out.permute(0, 2, 3, 1)
                        ).permute(0, 3, 1, 2).contiguous()
                    except Exception:
                        out = out.contiguous()
                features.append(out)
                continue

            if out.ndim == 3:
                if hasattr(self.model, "norm"):
                    try:
                        out = self.model.norm(out)
                    except Exception:
                        pass

                B = x.shape[0]
                patch_size = getattr(self, "patch_size", 14)
                if isinstance(patch_size, (tuple, list)):
                    patch_size = patch_size[0]

                gh = x.shape[2] // patch_size
                gw = x.shape[3] // patch_size
                need_tokens = gh * gw

                if out.shape[1] > need_tokens:
                    out = out[:, -need_tokens:, :]

                if out.shape[1] != need_tokens:
                    raise RuntimeError(
                        f"Cannot reshape DINO tokens: got {out.shape[1]} tokens, "
                        f"expected {need_tokens} = {gh}x{gw}."
                    )

                out = (
                    out.reshape(B, gh, gw, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                features.append(out)
                continue

            raise RuntimeError(f"Unsupported DINO output shape: {tuple(out.shape)}")

        return features
'''
    new_lines = lines[:start] + new_func.splitlines() + lines[end:]
    p.write_text("\n".join(new_lines) + "\n")
    py_compile.compile(str(p), doraise=True)
    print(f"[PATCHED] {p}")

def main():
    patch_droid_backends()
    patch_lietorch()
    patch_simple_knn()
    patch_numpy_2()
    patch_fit3d_feature_extractor()
    print("[DONE] DROID-W Colab patches applied")

if __name__ == "__main__":
    main()