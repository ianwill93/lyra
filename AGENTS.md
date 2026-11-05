# AGENTS.md

Guidance for AI agents working in this repository.

## Repository layout

This is the Project Lyra monorepo (NVIDIA). **Lyra 2.0 is the active project and all work
happens under `Lyra-2/`.** Lyra 1 was removed from the working tree (uncommitted deletions) —
do not resurrect it. Every Python/CMake command in this file runs from the `Lyra-2/` directory.

```
Lyra-2/
├── lyra_2/
│   ├── _src/                  # The actual project code (models, networks, tokenizers,
│   │                          #   datasets, inference, configs, callbacks, modules, utils)
│   ├── _ext/imaginaire/       # VENDORED NVIDIA Imaginaire training framework
│   │                          #   (config system, trainer, FSDP, DCP checkpointer).
│   │                          #   Copied from nvidia-cosmos/cosmos-predict2.5, with local
│   │                          #   patches. Treat as first-party, but don't restructure it.
│   └── _src/inference/{vipe, depth_anything_3}   # Git submodules with compiled CUDA exts
├── gui/                       # C++/CUDA real-time viewer (tiny-cuda-nn, CMake) + gui/api/
│   │                          #   Python FastAPI inference server
├── scripts/                   # train.py (training entry point), convert_wan_checkpoint.py
├── MoGe/                      # Git submodule (monocular depth), installed editable
├── checkpoints/               # Model weights (gitignored; HF download layout, see below)
├── outputs/                   # Run outputs (gitignored)
└── assets/                    # Sample inputs; assets/training_data/ = sample dataset (gitignored,
                               #   downloaded from HF, see TRAINING.md)
```

Submodules: `Lyra-2/.gitmodules` lists `vipe`, `depth_anything_3`, and ~9 `gui/dependencies/*`
(imgui, DLSS, OpenXR-SDK, pybind11, glfw, zlib, fmt, args, tinylogger). `MoGe` is a gitlink
**without** a `.gitmodules` entry (see "Blackwell / RTX PRO 6000"), so `git clone --recursive`
won't populate it — clone it manually. Use `gui/scripts/setup_dependencies.sh` for the GUI
deps.

## Environment

Tested on Ubuntu 22.04 + CUDA 12.8 on an NVIDIA **RTX PRO 6000 Blackwell Workstation
Edition** (96 GB, compute capability **12.0 / sm_120**). Conda env name is `lyra2`
(Python 3.10, torch 2.7.1+cu128, flash-attn 2.6.3 built from source, transformer-engine
1.13.0). The full agent-executable setup, including every Blackwell-specific fix and its
rationale, is **`ENV_SETUP.md`** (repo root). The upstream install reference is
`Lyra-2/INSTALL.md` — the order matters:
torch (cu128) → `pip install --no-deps -r requirements.txt` → MoGe (editable, pinned submodule
commit; install its git-backed extras separately) → `transformer_engine[pytorch]` (needs the
`nvidia/cuda_runtime` → `nvidia/cudart` symlink hack from INSTALL step 5) → flash-attn 2.6.3
built from source (`MAX_JOBS=4`) → editable CUDA builds of `vipe` and `depth_anything_3[gs]`
(`--no-build-isolation`, `USE_SYSTEM_EIGEN=1`).

Environment variables that must be set (details in INSTALL.md steps 4 and 5):
- `CPATH`, `LD_LIBRARY_PATH`, `CC`, `CXX` pointing into the conda env + torch site-packages
  (persist `LD_LIBRARY_PATH` in shell profile or imports will fail at runtime).
- `PYTHONPATH=.` for every `python` invocation — the `lyra_2` package is **not** pip-installed,
  it is imported from the repo root.
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before GPU runs.

Verify installation with the snippet in INSTALL.md step 8 (imports
`torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1`
and runs both inference scripts with `--help`).

## Blackwell / RTX PRO 6000

Everything in this section exists because the upstream tree targets H100 (sm_90) and
assumes Flash Attention 2 works on the GPU. The local target GPU is an NVIDIA RTX PRO
6000 Blackwell Workstation Edition (96 GB VRAM, **compute capability 12.0 / sm_120**,
driver 615.78.05, system CUDA 12.8 toolkit at `/usr/local/cuda-12.8`). Verified stack:
Python 3.10.20, torch 2.7.1+cu128, flash-attn 2.6.3 (source), transformer-engine 1.13.0,
cuDNN 9.7.1.26, NCCL 2.26.2.

**Core problem:** the Flash Attention 2 runtime in flash-attn 2.6.3 and Transformer
Engine's built-in FA2 kernel do not support sm_100+; cuDNN's native FMHA is the working
half-precision attention path on Blackwell. All code changes below route attention through
cuDNN and drop H100-only defenses that only hurt on a 96 GB card.

### Code changes (commit `2fa3918` "harsh adjustments to run on blackwell")

**Python / inference:**

- `lyra_2/_src/modules/attention.py` — new `elif compute_cap >= 100 and is_half:` branch:
  on SM100+ (covers sm_120) the SDPA backend priority is
  `[CUDNN_ATTENTION, EFFICIENT_ATTENTION, FLASH_ATTENTION]` with best = cuDNN. FA2 in
  flash-attn 2.6.3 predates Blackwell support, so it must not be preferred here.
- `lyra_2/_src/networks/wan2pt1.py` —
  - `SelfAttnOp` and `WanI2VCrossAttention.attn_op_image` now use
    `SafeDotProductAttention` instead of raw TE `DotProductAttention`. TE is only
    instantiated for compute capability 80–99; on Blackwell (or when TE raises the
    "FlashAttention only supports Ampere GPUs or newer" RuntimeError) it falls back to
    `generic_attention` (the cuDNN-preferring SDPA path above), and on CUDA OOM it falls
    back further to `_chunked_scaled_dot_product_attention` (chunk size 64, fp32
    accumulation, optional bf16/fp16 autocast).
  - `rope_apply`: `flash_attn` is now an optional import; when it is missing, a pure
    PyTorch interleaved rotary-embedding fallback is used instead of crashing.
- `lyra_2/_src/inference/lyra2_ar_inference.py` and `lyra2_zoomgs_inference.py` — cuDNN is
  now *enabled* + `benchmark=True` at module level and on the VAE decode path (upstream
  disabled it to dodge an H100 Conv3D workspace OOM; on this card that only forced slow
  fallback kernels). The `gc.collect()`/`torch.cuda.empty_cache()` calls after the VAE
handoff and at each AR-step boundary now run only when `self.args.offload or vae_offload`
is set, so the resident-GPU path stays hot.

**GUI (C++/CUDA):**

- `gui/CMakeLists.txt` — the CUDA-version → max-architecture mapping now goes
  `<12.0 → 90, <12.8 → 100, else → 120`, so a CUDA 12.8 toolkit targets sm_120 instead of
  dropping the detected architecture.
- `gui/dependencies/imguizmo/ImGuizmo.cpp` (6 sites) —
  `ImGui::SetNextFrameWantCaptureMouse(true)` → `ImGui::CaptureMouseFromApp(true)`; and
  `gui/src/python_api.cu` — `ImGuiMod_Alt/Ctrl/Shift/Super` → `ImGuiKeyModFlags_Alt/...`.
  The imgui submodule is pinned at v1.87-19-gfa2b318dd, which exposes the *old* API names;
  the upstream code assumed the newer ones and failed to compile.
- `gui/scripts/setup_dependencies.sh` — GUI submodules are now initialized from the
  monorepo superproject root (with `git submodule sync`); running `git submodule` from
  inside `Lyra-2` hits the nested `.gitmodules` layout and fails.

**Install / dependencies (`Lyra-2/INSTALL.md`):**

- MoGe: when a local `Lyra-2/MoGe/` checkout exists it is installed editable with
  `--no-deps`, and its git-backed extras are installed separately with pinned commits and
  `--no-deps` (`utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183`,
  `pipeline@866f059d2a05cde05e4a52211ec5051fd5f276d6`) so pip does not re-resolve torch.
  `Lyra-2/MoGe` was committed as a gitlink (microsoft/MoGe @ `925b8ed`) **without a
  `.gitmodules` entry** — a fresh clone will not populate it; clone it manually
  (ENV_SETUP.md, step 1).
- flash-attn 2.6.3: source build with `MAX_JOBS=4` (was 16; the CUDA build is
  memory-hungry and high parallelism was flaky on this box).

### Uncommitted local submodule patches (must survive any re-clone)

- `vipe` @ `b7cac64` — gdown 6.x dropped the `fuzzy=` kwarg; the three
  checkpoint-download call sites (`vipe/slam/networks/droid_net.py`,
  `vipe/priors/track_anything/__init__.py`, `scripts/download_dataset.py`) now use the
  `drive.google.com/uc?id=...` URL form, try `gdown.download(url, fuzzy=True)` and retry
  without `fuzzy` on `TypeError`.
- `depth_anything_3` @ `1ed6cb8` — `addict` added to `pyproject.toml` dependencies
  (imported but undeclared → ImportError after a clean install).
- `gui/dependencies/OpenXR-SDK` @ `e2da9ce` — `src/loader/CMakeLists.txt` adds
  `/usr/include` and `/usr/include/X11` to the loader include dirs (X11 dev headers live
  in system locations).

Exact, re-applicable patch text for all three is in `ENV_SETUP.md` step 8.

### Machine-level (outside the repo)

- `~/.bashrc`: `export CUDA_HOME=/usr/local/cuda-12.8` and
  `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH` — the live env uses
  the **system** CUDA 12.8 toolkit (nvcc 12.8.93), not the conda CUDA from INSTALL.md
  step 2 (there is no nvcc inside the conda env).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before GPU runs (see "Essential
  commands").

## Essential commands

All commands run from `Lyra-2/`.

### Inference (two-step pipeline)

Step 1 — generate the camera-controlled exploration video:

```bash
# Preset zoom-in/zoom-out from a single image (each sample image needs a sibling .txt caption)
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples --sample_id 4 \
  --experiment lyra2 --checkpoint_dir checkpoints/model \
  --prompt_dir assets/samples --output_path outputs/zoomgs \
  --num_frames_zoom_in 81 --num_frames_zoom_out 241

# Custom predefined trajectory (first_frame.png + trajectory.npz + captions.json)
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference \
  --input_image_path assets/custom_trajectory_examples/example_0/first_frame.png \
  --trajectory_path assets/custom_trajectory_examples/example_0/trajectory.npz \
  --experiment lyra2 --checkpoint_dir checkpoints/model \
  --captions_path assets/custom_trajectory_examples/example_0/captions.json \
  --num_frames 481 --output_path outputs/custom_traj
```

Step 2 — lift the video to a 3D Gaussian Splatting scene (VIPE pose + DA3 depth):

```bash
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon \
  --input_video_path outputs/zoomgs/videos/<sample_id>.mp4

# Long videos (>500 frames): chunk the DA3 pass, otherwise results blur
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_chunked_gs_recon \
  --input_video_path <video> --num_chunks 2 --chunk_size 128 --chunk_overlap 10
```

Fast mode: `--use_dmd` on either Step-1 command swaps in the 4-step DMD distillation LoRA
(`checkpoints/lora/dmd_distillation.safetensors`) plus its scheduler — ~15x faster but weaker
prompt following. Default (non-DMD) runs apply two LoRAs: `realism_boost` and `detail_enhancer`
(weight 0.4 each, see `--lora_paths`/`--lora_weights`).

### GUI (interactive)

Server (FastAPI, wraps the same pipeline; requires Qwen3-VL-4B at
`checkpoints/qwen/Qwen3-VL-4B-Instruct`):

```bash
export PYTHONPATH="$PWD:$PWD/gui/api"
python -m uvicorn server:app --app-dir gui/api --host 0.0.0.0 --port 8000
```

Client (C++/CUDA viewer built first; see `gui/README.md` for conde deps + CMake):
`python gui/api/client.py --host 127.0.0.1 --port 8000`.

### Training

Entry point is `scripts/train.py` (Hydra-style overrides go after `--`). Two-iteration smoke test
on one 8-GPU node (also the standard verification that a config resolves and visualizations work):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYRA2_DISABLE_CHECKPOINT_SAVE=1

torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=lyra_2/_src/configs/config.py -- \
  experiment=lyra2 \
  job.name=sample_training \
  trainer.max_iter=2 \
  trainer.logging_iter=1 \
  trainer.callbacks.frame_pack_viz_sampling.every_n=1 \
  trainer.callbacks.frame_pack_viz_sampling.num_sampling_step=1
```

Full-run and dataset-preparation details are in `TRAINING.md`. `--dryrun` on `scripts.train`
resolves the config and writes `config.yaml` under the job output dir without training.
`LYRA2_DISABLE_CHECKPOINT_SAVE=1` is a local trainer patch (see `lyra_2/_ext/imaginaire/trainer.py`).
Logs/TensorBoard/visualizations/checkpoints land in
`outputs/lyra_2/lyra2/<job.name>/{stdout.log,tensorboard/,visualizations/,checkpoints/}`.

### Checkpoints

HF download layout (from `README.md`): `huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .`
produces `checkpoints/{model,image_encoder,text_encoder,vae,recon,lora}/...`.
The main diffusion weights live at `checkpoints/model/model/*.distcp` — a DCP checkpoint sharded
across 64 FSDP ranks. `--checkpoint_dir checkpoints/model` is passed to
`lyra_2/_src/utils/model_loader.py`, which appends `/model` and `dcp.load`s with
`allow_partial_load=True` (reshardable to one GPU). A consolidated single `.pth` is also
supported by the same loader.

For training from scratch, the DiT is additionally initialized from the public Wan2.1 I2V 14B
480P weights: download the safetensors shards and convert with
`python -m scripts.convert_wan_checkpoint --input-dir ... --output .../Wan2.1-I2V-14B-480P.pth`
(needs ~45 GB CPU RAM, writes ~33 GB).

## Architecture

**Two-stage system.** (1) An autoregressive, camera-conditioned video diffusion model generates a
scene-walkthrough video; (2) feed-forward 3DGS reconstruction (VIPE SLAM poses + DA3 depth) lifts
it to a real-time-renderable scene. The two stages are completely separate code paths: Step 1 is
`lyra_2/_src/inference/lyra2_*_inference.py` + `models/`, Step 2 is
`lyra_2/_src/inference/vipe_da3_*_recon.py` + the `vipe`/`depth_anything_3` submodules.

**Generation model.** Backbone is the Wan 2.1 I2V 14B DiT
(`lyra_2/_src/networks/wan2pt1.py`, extended in `wan2pt1_lyra2.py`) wrapped by
`WANDiffusionModel` (`models/wan_t2v_model.py`, FSDP, LoRA injection, EMA) and
`Lyra2Model` (`models/lyra2_model.py`). Conditioning: T5 text embeddings **per 80-frame chunk**
(not one prompt for the whole video), CLIP image embedding of the first frame
(`modules/clip.py`, `networks/clip_lyra2.py`), and camera controls via Plücker embeddings + a
camera encoder (`modules/conditioner.py`). The VAE is the Wan VAE
(`tokenizers/wan2pt1.py`) used in a *streaming/cached* mode: encode/decode proceed chunk-by-chunk
over `_enc_feat_map`/`_dec_feat_map` feature caches; the model exposes
`_vae_encode_range_stream`, `vae_encode_with_cache`, `_clone_vae_cache`.

**Autoregressive "framepack" chunking.** Generation proceeds in 80-frame chunks; the total frame
count must be of the form 1 + 80k. The layout of history/new tokens is encoded in the
`framepack_type` config string (e.g. `f1k1f4s2f1s1f16k4f2k2f1k1_g20` in
`lyra_2/_src/configs/experiment.py`) together with `max_segments`. `Sparse3DCache`
(`models/lyra2_model.py`) stores per-frame RGB/depth for spatial memory, and
`datasets/forward_warp_utils_pytorch.py` provides the forward-warp correspondence used to route
past-frame information. Training additionally uses *self-augmented histories*
(`self_aug_*` flags in `experiment.py`): the model is trained on its own degraded outputs to
correct drift.

**Inference engine.** `Lyra2InferencePipeline` in
`lyra_2/_src/inference/lyra2_ar_inference.py` is the single-GPU driver shared by all three
inference entry points and the GUI server. It is deliberately memory-aggressive: the DiT is
offloaded to CPU around every VAE operation (`_offload_diffusion_to_cpu`/`_restore_diffusion_to_gpu`),
the VAE decoder can run layer-wise offloaded, and DA3 depth is windowed
(`_predict_da3_depth_window`). It also supports `save_snapshot`/`revert_to_snapshot` for the
GUI's branch-and-revert interaction. `torch.enable_grad(False)` is set at module import time in
the inference scripts; cuDNN is explicitly enabled + `benchmark=True` for the VAE decode path
(local Blackwell adjustment, see below).

**Training framework.** `lyra_2/_ext/imaginaire` (vendored) provides the config system: Python
`attrs` Configs + Hydra defaults + `LazyCall` (`L()`) instantiation.
`lyra_2/_src/configs/config.py::make_config()` builds the root config and registers the defaults
groups; named experiments are registered via Hydra `ConfigStore` in
`lyra_2/_src/configs/experiment.py` (currently one: `lyra2`). To point training at a new dataset:
add an entry to `DATAVERSE_CONFIG` in `lyra_2/_src/datasets/config_dataverse.py` (every entry is
auto-registered) and override `/data_train` and `/data_val` in `experiment.py`. The dataloader
chain is `DepthWarpDataset` → `InfiniteCommonDataset` (`datasets/depth_warp_dataloader.py`),
backed by the `Radym` dataset class (`datasets/radym.py`) over the dataset layout documented in
TRAINING.md (per-video-id `rgb/`, `pose/`, `intrinsics/`, `depth/` + chunk-caption `.pkl` files
containing `frame_range`, `caption`, and bf16 T5 `embedding`).

**GUI.** `gui/api/server_lyra.py` (`LyraModel`) wraps the inference pipeline behind a FastAPI app
with an allowlisted JSON protocol; `qwen_captioner.py` captions frames with Qwen3-VL for the
autoregressive extension loop. `gui/src/*.cu` + `gui/include/` is the C++/CUDA viewer (built with
CMake, pybind11 bindings, DLSS/OpenXR optional) driven from `gui/api/client.py`.

## Conventions and gotchas

- **Always run from `Lyra-2/` with `PYTHONPATH=.`** — module imports of the form
  `lyra_2._src....` resolve only that way.
- **No test suite in the main `lyra_2` code** (pytest is in requirements; the only test files
  live inside submodules). The de-facto verification workflow is: the 2-iteration training smoke
  test above, `--help` on the inference scripts, and a sample zoomgs run.
- **Local divergence from upstream.** This branch is one commit ahead of
  `nv-tlabs/lyra`: `2fa3918` — the complete list of Blackwell changes, with rationale,
  is in the "Blackwell / RTX PRO 6000" section above and in `ENV_SETUP.md`. Be careful
  rebasing/merging from upstream.
- **Uncommitted local patches inside submodules** (shown as `m` in `git status`): `vipe`
  (gdown API compatibility in `droid_net.py`, `track_anything/__init__.py`,
  `download_dataset.py`), `depth_anything_3` (`addict` added to `pyproject.toml` deps),
  `gui/dependencies/OpenXR-SDK` (`src/loader/CMakeLists.txt` X11 include dirs). Repro
  patch text is in `ENV_SETUP.md` step 8. **Do not run
  `git submodule update --force`** — it would wipe these compatibility fixes.
- **`checkpoints/`, `outputs/`, `assets/training_data/`, `cam.json` are local/gitignored.**
  `Lyra-2/cam.json` is untracked GUI camera-path runtime state, not part of the repo.
- **Frame-count rule:** generated videos must be 1 + 80k frames (`(N-1)` divisible by
  `tokens_per_step * frames_per_latent`), enforced by an assert in `run_lyra2_sample`.
- **Zoom behavior:** `--zoom_in_strength`/`--zoom_out_strength` may have no visible effect because
  a built-in collision detector clamps the trajectory.
- **Custom trajectories:** the supplied trajectory is re-scaled relative to monocular depth of
  the first frame; adjust with `--pose_scale` if the path looks wrong.
- **DA3 recon quality:** single-pass DA3 degrades past ~500 frames; use the chunked recon script.
- **LoRA defaults:** non-DMD runs silently apply `realism_boost` + `detail_enhancer` LoRAs at
  weight 0.4 unless `--lora_paths` is overridden; `--use_dmd` replaces them with the DMD LoRA.
- **Training outputs:** default `job.name` is `delete_<timestamp>` — set `job.name` explicitly or
  job dirs become trash. `checkpoint.save_iter=100` is the release cadence.
- **New experiments:** register via `ConfigStore` (group `experiment`, `package="_global_"`) in
  `lyra_2/_src/configs/experiment.py`; the `defaults` list with `override /...` entries selects
  model/net/conditioner/data/ckpt combinations.
- **Commits:** upstream requires DCO sign-off — use `git commit -s` (see `CONTRIBUTING.md`).
- **Licenses:** code is Apache-2.0; the released model weights are under NVIDIA's internal
  scientific research license (not MIT/Apache).
- **Vendored code discipline:** `lyra_2/_ext/imaginaire` and the submodule packages are
  vendored/copied code with upstreams (cosmos-predict2.5, nv-tlabs/vipe,
  Depth-Anything-3 fork, microsoft/MoGe). Fix bugs in-place where release needs it, but keep
  patches minimal and local so they remain easy to diff against upstream.
