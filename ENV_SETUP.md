# ENV_SETUP.md — Preparing a Linux environment for Lyra 2 (Blackwell / RTX PRO 6000)

Step-by-step, agent-executable procedure to take a fresh Linux machine or container from
scratch to a working Lyra 2 installation on a Blackwell-class GPU. Verified on an
**NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (96 GB, compute capability **12.0 /
sm_120**). It includes the uncommitted Blackwell compatibility patches that live only in
local working trees and would be lost on a plain re-clone.

- The *why* behind each Blackwell fix: `AGENTS.md`, section "Blackwell / RTX PRO 6000".
- Upstream (H100-oriented) install reference: `Lyra-2/INSTALL.md`. This document supersedes
  it for Blackwell.

All commands are run from the repository root unless stated otherwise. Every step ends
with a **verify** line — run it and only continue on success.

## Verified reference environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 x86_64 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM |
| Compute capability | 12.0 (sm_120); `torch.cuda.get_device_capability(0)` → `(12, 0)` |
| Driver | 615.78.05 (any driver supporting CUDA 12.8 + Blackwell works; ≥570.x) |
| CUDA toolkit | 12.8, system location `/usr/local/cuda-12.8` (nvcc 12.8.93) |
| Conda env | `lyra2`, Python 3.10.20 |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| flash-attn | 2.6.3, built from source |
| transformer-engine | 1.13.0 |
| cuDNN / NCCL | 9.7.1.26 / 2.26.2 (pip `nvidia-*` wheels shipped with torch) |
| gdown | 6.1.0 (new gdown is why the vipe patch in step 8 is required) |
| Disk | ~100 GB free minimum (env + builds + checkpoints) |

## Step 0 — System packages and CUDA toolkit

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build pkg-config \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libxkbcommon-dev libgl1-mesa-dev libglew-dev
```

Install Miniconda to `~/miniconda3` and the CUDA **12.8** toolkit so that
`/usr/local/cuda-12.8/bin/nvcc` exists (NVIDIA apt repo, or use a container image that
already ships it, e.g. `nvidia/cuda:12.8.0-devel-ubuntu24.04`).

**Verify:**

```bash
nvcc --version          # release 12.8
nvidia-smi              # shows the Blackwell GPU; driver ≥570.x
```

## Step 1 — Get the code, submodules, and the MoGe quirk

```bash
git clone --recursive https://github.com/ianwill93/lyra.git
cd lyra
```

The clone prints a warning like `No url found for submodule path 'Lyra-2/MoGe'` —
**expected, non-fatal**: `Lyra-2/MoGe` was committed as a gitlink without a `.gitmodules`
entry. Populate it manually:

```bash
git clone https://github.com/microsoft/MoGe.git Lyra-2/MoGe
git -C Lyra-2/MoGe checkout 925b8ed835a7a9cdb7578ba15c658a0afc969030
```

**Verify:**

```bash
git -C Lyra-2/lyra_2/_src/inference/vipe log --oneline -1            # b7cac64 ...
git -C Lyra-2/lyra_2/_src/inference/depth_anything_3 log --oneline -1 # 1ed6cb8 ...
git -C Lyra-2/MoGe log --oneline -1                                   # 925b8ed ...
git -C Lyra-2/gui/dependencies/imgui log --oneline -1                 # fa2b318dd ... (v1.87-19)
git -C Lyra-2/gui/dependencies/OpenXR-SDK log --oneline -1            # e2da9ce ...
```

If any pin differs, check it out explicitly before continuing.

## Step 2 — Conda environment

```bash
conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
conda activate lyra2
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y
```

Note: unlike `Lyra-2/INSTALL.md` step 2, this procedure uses the **system** CUDA toolkit
from step 0, so `conda install cuda -c nvidia/label/cuda-12.8.0` is intentionally skipped.
(Using conda CUDA instead is acceptable; in that case set `CUDA_HOME=$CONDA_PREFIX` in
step 4.)

**Verify:**

```bash
python -V   # Python 3.10.x
$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ --version | head -1   # gcc 13.3.0
```

## Step 3 — PyTorch (cu128)

```bash
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

**Verify:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 2.7.1+cu128 True (12, 0)
```

## Step 4 — Build environment variables

Run in every shell that compiles or runs anything:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Persist** `CUDA_HOME` and `LD_LIBRARY_PATH` in `~/.bashrc` (or as `ENV` lines in a
Dockerfile — see Appendix C); without the persisted `LD_LIBRARY_PATH`, imports of
`torch`/`flash_attn`/`transformer_engine` fail at runtime with missing-library errors.

## Step 5 — Python dependencies

```bash
cd Lyra-2
pip install --no-deps -r requirements.txt

# MoGe: install the local checkout (step 1) editable, then its git-backed extras
# separately so pip does not re-resolve torch.
pip install --no-deps -e "${PWD}/MoGe"
pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
pip install --no-deps "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"

# Transformer Engine + the cudart symlink hack it needs
pip install --no-build-isolation "transformer_engine[pytorch]"
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
```

The symlink exists because `transformer_engine` looks for `nvidia/cudart` while the pip
wheel ships `nvidia/cuda_runtime`.

**Verify:**

```bash
python -c "import transformer_engine.pytorch; print('TE ok')"
python -c "import moge.model.v1, utils3d, pipeline; print('MoGe stack ok')"
```

## Step 6 — Flash Attention (source build, ~30–90 min)

```bash
MAX_JOBS=4 pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
```

Keep `MAX_JOBS=4` — the CUDA build is memory-hungry and higher parallelism was flaky on
the reference machine. This step can run in the background; do nothing else GPU-heavy
while it compiles.

**Verify:**

```bash
python -c "import flash_attn; print(flash_attn.__version__)"   # 2.6.3
```

Importing `flash_attn` must not raise. Note: on sm_100+ the runtime *prefers* cuDNN SDPA
for attention (step 8 code); flash-attn still must be importable because the code imports
it at module level (and uses its rotary embedding when available).

## Step 7 — Vendored CUDA extensions

```bash
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
```

`USE_SYSTEM_EIGEN=1` makes vipe use the conda-installed Eigen instead of fetching a
private copy. Both builds compile CUDA kernels for the detected GPU (sm_120 here) and
take a while.

**Verify:**

```bash
python -c "import vipe_ext, depth_anything_3.api; print('CUDA exts ok')"
```

## Step 8 — Blackwell code state and local submodule patches

### 8a. Confirm the Blackwell commit is present

```bash
git merge-base --is-ancestor 2fa3918eb65d6423cbbf78d6fa764d05b79f5c4d HEAD && echo "blackwell commit present"
```

Spot-check the key markers (all must print a match):

```bash
grep -n "compute_cap >= 100" Lyra-2/lyra_2/_src/modules/attention.py
grep -n "class SafeDotProductAttention" Lyra-2/lyra_2/_src/networks/wan2pt1.py
grep -n "cudnn.enabled = True" Lyra-2/lyra_2/_src/inference/lyra2_ar_inference.py \
  Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py
grep -n "set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 120)" Lyra-2/gui/CMakeLists.txt
```

If the commit is missing (e.g. you cloned `nv-tlabs/lyra` upstream), fetch it from the
fork: `git fetch https://github.com/ianwill93/lyra.git main && git checkout FETCH_HEAD`.

### 8b. vipe: gdown 6.x compatibility (3 files)

New gdown (≥6) removed the `fuzzy=` keyword; the original call sites raise
`TypeError` when downloading the DROID/AOT checkpoints. In each of the three files
replace the shown call with the patched form.

`Lyra-2/lyra_2/_src/inference/vipe/vipe/slam/networks/droid_net.py` (~line 534):

```python
# before:
            gdown.download(
                "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view",
                output=str(ckpt_path),
                fuzzy=True,
            )
# after:
            url = "https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"
            try:
                gdown.download(url, output=str(ckpt_path), fuzzy=True)
            except TypeError:
                gdown.download(url, output=str(ckpt_path))
```

`Lyra-2/lyra_2/_src/inference/vipe/vipe/priors/track_anything/__init__.py` (~line 32):

```python
# before:
            gdown.download(
                "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view",
                output=str(aot_ckpt_path),
                fuzzy=True,
            )
# after:
            url = "https://drive.google.com/uc?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"
            try:
                gdown.download(url, output=str(aot_ckpt_path), fuzzy=True)
            except TypeError:
                gdown.download(url, output=str(aot_ckpt_path))
```

`Lyra-2/lyra_2/_src/inference/vipe/scripts/download_dataset.py` (~line 144):

```python
# before:
                    gdown.download(
                        "https://drive.google.com/file/d/1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs/view",
                        output=str(output_base / "web360_raw.zip"),
                        fuzzy=True,
                        use_cookies=False,
                    )
# after:
                    url = "https://drive.google.com/uc?id=1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs"
                    try:
                        gdown.download(url, output=str(output_base / "web360_raw.zip"), fuzzy=True, use_cookies=False)
                    except TypeError:
                        gdown.download(url, output=str(output_base / "web360_raw.zip"), use_cookies=False)
```

### 8c. depth_anything_3: declare missing `addict` dependency

In `Lyra-2/lyra_2/_src/inference/depth_anything_3/pyproject.toml`, in the
`dependencies` list, add `"addict",` immediately after the `"pillow_heif",` line.
Without it, a clean install misses the package and DA3 raises `ImportError: addict`.

### 8d. OpenXR-SDK: X11 include dirs (GUI only)

In `Lyra-2/gui/dependencies/OpenXR-SDK/src/loader/CMakeLists.txt`, inside the `else()`
branch, extend the loader include dirs:

```cmake
    target_include_directories(openxr_loader
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/external/jsoncpp/include
        /usr/include
        /usr/include/X11
    )
```

(Only needed if you build the GUI with OpenXR; harmless otherwise.)

### 8e. lyra2_ar_inference: survive a diverged chunk (DA3 pose guard)

A real web-UI generation died ~40 min into its first chunk with

```
torch._C._LinAlgError: linalg.inv: The diagonal element 1 is zero, the
inversion could not be completed because the input matrix is singular.
  lyra2_ar_inference.py:1151  inv_p0 = torch.linalg.inv(pred_w2c_all[hist_local[0]])
```

The container log names the cause one step earlier: `RuntimeWarning: invalid value
encountered in cast` at `lyra2_ar_inference.py:326`. A diverged diffusion chunk leaves
NaN/Inf in `history_frames`; `clamp()` propagates NaN, the uint8 cast hands DA3 an
arbitrary frame, and DA3 answers with a rotation that has a zero column. Because this runs
inside the AR cache update, one bad matrix aborted the whole chunk and
`revert_to_snapshot()` threw the world away, so the user saw a bare 500 after 40 minutes.

Apply the guard (idempotent; asserts every anchor matches exactly once, byte-compiles after):

```bash
python "/mnt/f/VMs/WSLC - Builds/build-lyra2/scripts/patch_lyra2_da3_pose_guard.py" \
  /home/ianwi/lyra/Lyra-2
# patched lyra2_ar_inference.py: applied=[...] already-present=none
```

Four changes in `lyra_2/_src/inference/lyra2_ar_inference.py`:

1. `_predict_da3_depth_window`: `np.nan_to_num` before the uint8 cast plus one
   `[DA3] N of M window frame(s) had non-finite pixels ...` warning.
2. `autoregressive_step`: `[NaN] AR step k generated non-finite pixels in N frame(s) [...]`,
   naming the frames where they appear (and the usual cause: authored cameras too far from
   the seeded world, or too close together).
3. New `_poses_are_invertible(w2c_all)`: finite, and every rotation determinant ~1.
4. Both predicted-pose alignment branches gated on it, so a degenerate DA3 answer degrades
   to the pipeline poses - the fallback the code already has for "DA3 returned no
   extrinsics" - instead of failing the run.

**Verify (real GPU):** generate a chunk on a deliberately bad camera path (e.g. keyframes
authored 60 units from the seed). The run must finish with `[NaN] AR step ...`,
`[DA3] ...` and `[da3_use_predicted_pose] ... keeping the pipeline poses` in the log and a
written `chunk_0000_*.mp4`, instead of the `linalg.inv` traceback.

**Verify:** a fresh checkout with these patches applied must show exactly this in
`git status`:

```bash
git -C Lyra-2/lyra_2/_src/inference/vipe status --short
#  M scripts/download_dataset.py
#  M vipe/priors/track_anything/__init__.py
#  M vipe/slam/networks/droid_net.py
git -C Lyra-2/lyra_2/_src/inference/depth_anything_3 status --short
#  M pyproject.toml
git -C Lyra-2/gui/dependencies/OpenXR-SDK status --short
#  M src/loader/CMakeLists.txt
```

**Never run `git submodule update --force`** in this repo — it wipes these patches.

## Step 9 — GUI build (optional; skip for headless/container server-only use)

```bash
conda install -c conda-forge -y glew libgl-devel xorg-libx11 xorg-libxrandr \
  xorg-libxinerama xorg-libxcursor xorg-libxi pkg-config
bash gui/scripts/setup_dependencies.sh
pip install -r gui/requirements.txt
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
cmake gui -B gui/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON
cmake --build gui/build --config RelWithDebInfo -j
```

Notes:

- `setup_dependencies.sh` initializes the GUI submodules from the monorepo root; it
  errors if `Lyra-2/.gitmodules` is missing (i.e. if you built the tree from a tarball
  instead of a git checkout, check out the dependencies manually).
- The CMakeLists fix in the Blackwell commit makes CUDA 12.8 auto-target sm_120; no
  `TCNN_CUDA_ARCHITECTURES` override is needed.
- Requires the OpenXR X11 patch from step 8d when OpenXR is enabled.
- The C++ *viewer* needs a display; in a container run the Python *server* only (step 10).

## Step 10 — Install checkpoints and verify

```bash
cd Lyra-2
pip install -U huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
# layout: checkpoints/{model,image_encoder,text_encoder,vae,recon,lora}/...
# main diffusion weights: checkpoints/model/model/*.distcp (DCP, ~35 GB)
```

Final verification (run from `Lyra-2/`):

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```

Optional end-to-end smoke test (needs the checkpoints from above):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples --sample_id 4 \
  --experiment lyra2 --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples --output_path outputs/zoomgs \
  --num_frames_zoom_in 81 --num_frames_zoom_out 241
```

On Blackwell, attention runs through the cuDNN SDPA backend and VAE decode through
cuDNN convs (both enabled + benchmarked). If you see "FlashAttention only supports
Ampere GPUs or newer" errors or extremely slow diffusion steps, re-check step 8.

## Appendix A — Environment variables to persist

| Variable | Value | Where |
|---|---|---|
| `CUDA_HOME` | `/usr/local/cuda-12.8` | shell profile / Dockerfile `ENV` |
| `LD_LIBRARY_PATH` | `$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64` | shell profile / Dockerfile `ENV` |
| `CPATH` | `$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include` | build shells only |
| `CC` / `CXX` | `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g{cc,++}` | build shells only |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | every GPU run |
| `PYTHONPATH` | `.` (run from `Lyra-2/`; GUI server adds `:$PWD/gui/api`) | every `python` invocation |

`SITE=$CONDA_PREFIX/lib/python3.10/site-packages`. The `lyra_2` package is not
pip-installed — it is imported from the repo root, hence `PYTHONPATH=.`.

## Appendix B — Blackwell fixes quick reference

| File | Change | Why |
|---|---|---|
| `lyra_2/_src/modules/attention.py` | `compute_cap >= 100` half-precision → cuDNN-first SDPA backend order | FA2 (flash-attn 2.6.3) does not support sm_100+; cuDNN FMHA is the stable path |
| `lyra_2/_src/networks/wan2pt1.py` | `SafeDotProductAttention` wraps TE (cc 80–99 only) → `generic_attention` → chunked SDPA on OOM; `rope_apply` pure-PyTorch fallback; optional `flash_attn` import | TE's built-in FA2 kernel refuses sm_120; rope must not crash without flash-attn |
| `lyra_2/_src/inference/lyra2_ar_inference.py`, `lyra2_zoomgs_inference.py` | cuDNN enabled + `benchmark=True` (module-level and VAE decode); offload-gated `gc.collect()`/`empty_cache()` | H100-only cuDNN-off defense forced slow fallback kernels on the 96 GB card; cache flushes hurt the resident-GPU path |
| `gui/CMakeLists.txt` | CUDA-version → max-arch mapping `<12.0→90, <12.8→100, else→120` | CUDA 12.8 can target sm_120; upstream stopped at 90 |
| `gui/dependencies/imguizmo/ImGuizmo.cpp`, `gui/src/python_api.cu` | `CaptureMouseFromApp(true)`, `ImGuiKeyModFlags_*` | Pinned imgui (v1.87-19) exposes the old API names; upstream assumed newer ones |
| `gui/scripts/setup_dependencies.sh` | Submodule init from superproject root | Nested `.gitmodules` layout in the monorepo |
| `gui/dependencies/OpenXR-SDK` (local patch) | `+ /usr/include, /usr/include/X11` include dirs | X11 dev headers live in system locations |
| `Lyra-2/INSTALL.md` | MoGe editable + pinned `--no-deps` extras; flash-attn `MAX_JOBS=4` | Avoid torch re-resolution; stable flash-attn build |
| `Lyra-2/MoGe` | Gitlink @ `925b8ed`, **no `.gitmodules` entry** | Fresh clones must clone it manually (step 1) |
| `vipe` (local patch ×3) | gdown 6.x `fuzzy=` retry without keyword | gdown ≥6 raises `TypeError` on the old call form |
| `depth_anything_3` (local patch) | `addict` added to `pyproject.toml` deps | Undeclared import → `ImportError` after clean install |

## Appendix C — Notes for WSLC / containerization

- **Driver:** provided by the Windows host. The host NVIDIA driver must support
  Blackwell + CUDA 12.8 (≥570.x; verified 615.78.05). Never install driver packages in
  the container; attach the GPU via the NVIDIA container runtime (`--gpus all` /
  `nvidia` device).
- **Base image:** `nvidia/cuda:12.8.0-devel-ubuntu24.04` gives you `/usr/local/cuda`
  pointing at 12.8; either `ln -s /usr/local/cuda /usr/local/cuda-12.8` or set
  `CUDA_HOME=/usr/local/cuda` so step 4's paths stay valid.
- **Env vars:** bake Appendix A into the Dockerfile as `ENV` lines (the `~/.bashrc`
  persistence trick does not apply to non-interactive containers).
- **Build layering:** steps 2–7 (conda env + source builds) are slow — especially the
  flash-attn build (step 6). Bake them into image layers *before* copying the
  repository, so code changes don't invalidate the compiled environment.
- **Scope:** the container should run the inference stack (steps 0–8, 10). The C++ GUI
  *viewer* needs a display — run it on the host/WSL desktop against the containerized
  server (`python gui/api/client.py --host ... --port 8000`), or skip step 9 entirely.
- **Server entrypoint** (repo assumed checked out at `/app`; adjust paths accordingly):
  `PYTHONPATH=/app/Lyra-2:/app/Lyra-2/gui/api python -m uvicorn server:app
  --app-dir gui/api --host 0.0.0.0 --port 8000` (no systemd in WSLC).
- **State:** `checkpoints/` (~35 GB+), `outputs/`, and `Lyra-2/cam.json` are local and
  gitignored — mount them as volumes so they survive image rebuilds.
- **Re-applying patches:** after any re-clone or `git submodule update` (never
  `--force`), re-run step 8b–8d; the 8b–8d text is the canonical patch spec.
