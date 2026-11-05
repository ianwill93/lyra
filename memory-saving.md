# VRAM / Memory Saving Research — Lyra 2.0

Research notes on reducing VRAM for single-GPU inference (goal: run in a container on
smaller cards). Findings as of 2026-10-07, branch `main` (commit `2fa3918` + local patches).

## 1. No quantization support today

- Nothing in the repo references torchao, bitsandbytes, FP8, or int8 inference paths
  (grep hits are false positives: quaternions, quantiles, int8 depth arrays).
- All components are loaded and run in bf16:
  - DiT precision is fixed at `precision: str = "bfloat16"` —
    `lyra_2/_src/models/wan_t2v_model.py:110`, applied via `self.net.to(**self.tensor_kwargs)`
    in `on_train_start` (`wan_t2v_model.py:325`).
  - `lyra_2/_src/utils/model_loader.py:70` does `instantiate(config.model).cuda()`
    straight away — the full bf16 DiT lands on GPU before anything else, so even quantizing
    after load pays the bf16 GPU spike.

## 2. Current VRAM breakdown (step 1, `lyra2_zoomgs_inference.py`)

Measured from `checkpoints/` dir sizes + code paths:

| Component | On-disk size | GPU footprint (bf16/fp16) | Where loaded |
|---|---|---|---|
| DiT Wan 2.1 I2V 14B | 64 GB DCP shards (`checkpoints/model`) | ~28 GB | `model_loader.load_model_from_checkpoint` |
| UMT5-xxl text encoder | 11 GB (`checkpoints/text_encoder`) | ~11 GB | `get_t5_emb.UMT5EncoderModel` (`get_t5_emb.py:492`), resident on GPU |
| DA3NESTED-GIANT-LARGE | 13 GB (`checkpoints/recon`) | ~13 GB | `lyra2_zoomgs_inference.py:560` — loaded **before** generation, stays resident; used only for windowed depth of generated frames in the AR loop (`_predict_da3_depth_window`) and first-frame depth |
| CLIP ViT-H/14 (xlm-roberta) | 2.3 GB (`checkpoints/image_encoder`) | ~2.3 GB fp16 | `modules/clip.py:493 CLIPModel`; used once for the first frame |
| Wan VAE | 485 MB (`checkpoints/vae`) | ~1 GB bf16 | `tokenizers/wan2pt1.py` (`dtype=torch.bfloat16, is_amp=False`) |
| LoRA adapters (realism_boost / detail_enhancer / DMD) | 1.1 GB (`checkpoints/lora`) | small each | PEFT `inject_adapter_in_model` (`wan_t2v_model.py:466`) |
| Qwen3-VL-4B (GUI server only) | 8.3 GB (`checkpoints/qwen`) | ~8 GB | `gui/api/qwen_captioner.py` |

~55 GB of resident weights in step 1 (before activations) → H100 80 GB / RTX PRO 6000 96 GB
required. Note: generation and 3DGS recon are separate processes/scripts, so DA3 in the
recon script does not coexist with the DiT there; the co-residency problem is inside
`lyra2_zoomgs_inference.py` / `lyra2_custom_traj_inference.py` only.

## 3. The obvious quantization path: torchao on the DiT

The DiT is a stock `nn.Linear` transformer
(`lyra_2/_src/networks/wan2pt1.py`: `WanSelfAttention`, `WanAttentionBlock.ffn`, `Head`,
`patch_embedding`, `text_embedding`, `time_embedding` — all plain `nn.Linear`), and on the
single-GPU inference path there is **no FSDP wrap**: `enable_fsdp=False` →
`fsdp_shard_size=1` → the `fully_shard` branches (`wan_t2v_model.py:249`,
`lyra2_model.py:281`) never run. This is exactly the module structure
`torchao.quantization.quantize_` targets.

Options (in recommended order):

1. **`Float8WeightsOnlyLinearConfig()`** — fp8 e4m3 weights, 28 → ~14 GB. Best quality;
   on sm_120 Blackwell FP8 GEMM is native via cuBLAS `scaled_mm` (sm_89+), so the speed
   hit is minimal. Try first.
2. **`int8_weight_only()`** — 28 → ~14 GB, int8 tensor cores (sm_80+), slightly more
   aggressive, fine on CPU too.
3. **`int4_weight_only()`** — 28 → ~8–9 GB, biggest savings, real quality risk for a
   detail-critical video model: artifacts get baked into the 3DGS scene in step 2.

Implementation notes / gotchas:

- Apply to `model.net` right after `load_model_from_checkpoint`
  (`lyra2_zoomgs_inference.py:~518`, and the same spot in `lyra2_custom_traj_inference.py`
  and `gui/api/server_lyra.py`).
- **PEFT gotcha:** LoRAs are injected via `inject_adapter_in_model`, which wraps each
  `nn.Linear` in `peft.lora.Linear`; torchao's default `isinstance(module, nn.Linear)`
  filter will not match it. Use a custom `filter_fn` that matches `.base_layer`
  (or quantize `base_layer` directly / quantize before injection). Verify quantization
  survives `load_lora_weights` + `set_weights_and_activate_adapters`
  (`wan_t2v_model.py:353`).
- **CPU-load path required for small cards:** `model_loader.py:70` moves the model to
  GPU in bf16 first. To actually run on a 24 GB card: instantiate on CPU, `dcp.load` into
  CPU tensors (~28 GB RAM), `quantize_` on CPU, then `.to(cuda)`. (Weight-only torchao
  configs support CPU-side quantization; FP8 GEMM path is GPU-side, so quantize fp8 on
  CPU but the runtime `scaled_mm` needs GPU.)
- **Offline pre-quantization (best for containers):** quantize the DCP checkpoint once
  into a single `.pth` of quantized weights (torchao saves/loads quantized state dicts;
  `model_loader.py:74` already accepts a consolidated `.pth`). Runtime then loads the
  quantized weights directly — no bf16 spike, faster container builds.
- Add `torchao` to `requirements.txt` (not present today).
- Quality caveat: `--use_dmd` (4-step distillation) + quantization is a double hit;
  validate with full-step counts first.

## 4. Zero-risk wins (no quantization needed)

- **`--offload_da3_diffusion`** already exists (`lyra2_ar_inference.py:1077`) — moves the
  13 GB DA3 to CPU during diffusion. Enable by default; alone saves ~13 GB of resident
  VRAM.
- **`--offload_when_prompt`** (`lyra2_zoomgs_inference.py:694`) — runs UMT5 on CPU
  (`get_umt5_embedding_offloaded`, `get_t5_emb.py:556`). For batch/offline runs the chunk
  captions are known up front, so all T5 embeddings could be precomputed and the 11 GB
  encoder dropped entirely.
- **CLIP:** used once for the first-frame encode (`clip_lyra2.py:49`); move to CPU after
  use (or precompute the embedding).

Impact: ~55 GB → ~33 GB resident weights with flags alone; combined with fp8/int8 DiT
quantization → ~17–20 GB total, fitting a 24 GB card with headroom for activations.

## 5. Suggested plan

1. Add `torchao` to `requirements.txt`.
2. Add `--quantize {none,fp8,int8,int4}` CLI flag; apply `quantize_` to `model.net`
   after checkpoint load, with a PEFT-aware `filter_fn`.
3. CPU-load path (instantiate on CPU → dcp.load on CPU → quantize → `.cuda()`) gated on
   quantization or a new `--load_on_cpu` flag.
4. Default `--offload_da3_diffusion` + `--offload_when_prompt` on.
5. Optional: offline pre-quantized `.pth` DiT for fast container startup.
6. Verify: run a short zoomgs sample, compare PSNR/visual quality vs bf16 baseline;
   check LoRAs still active after quantization; watch for fp8 GEMM falling back to slow
   kernels (profile the DiT step).

## 6. References (file:line)

- `lyra_2/_src/models/wan_t2v_model.py:110` — `precision: str = "bfloat16"`
- `lyra_2/_src/models/wan_t2v_model.py:320-325` — `on_train_start` dtype cast
- `lyra_2/_src/models/wan_t2v_model.py:457-466` — PEFT LoRA injection
- `lyra_2/_src/utils/model_loader.py:66-92` — FSDP-disable + DCP/.pth load
- `lyra_2/_src/networks/wan2pt1.py:403-660` — DiT blocks (plain nn.Linear)
- `lyra_2/_src/inference/get_t5_emb.py:441-540` — UMT5-xxl encoder (bf16 default)
- `lyra_2/_src/modules/clip.py:493-540` — CLIP fp16
- `lyra_2/_src/tokenizers/wan2pt1.py:898-901` — VAE bf16, no AMP
- `lyra_2/_src/inference/lyra2_zoomgs_inference.py:505-575` — model/T5/DA3 load order
- `lyra_2/_src/inference/lyra2_zoomgs_inference.py:179-235` — existing offload flags
- `lyra_2/_src/inference/lyra2_ar_inference.py:1077` — `offload_da3_diffusion`
- `lyra_2/_src/inference/depth_utils.py:197-215` — `load_da3_model`
