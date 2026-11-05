# PLAN-ONE-LLM.md — one LLM endpoint in front of every world model

Created 2026-10-15. Companion to `PLAN.md` (HY-World 2.0), `PLAN-LYRA2.md` /
`PLAN-LYRA2-MODELSWITCH.md` (Lyra 2), `PLAN-MODEL-AGNOSTIC.md` (the shell).

**Scope:** the generative chat/vision model that world models call for scene
understanding. **Not** the UMT5-xxl text *encoders* (see §1, last row) — those stay
in-container and are not LLMs.

---

## 0. The idea in three sentences

Every world model gets its language/vision generation from **one OpenAI-compatible
`/v1` endpoint**, configured in exactly one place. Which engine serves it is a
deployment detail: **Foundry Local on this machine today**, vLLM or SGLang on a
server later — the app and the containers never learn the difference. A future
fine-tuned "WorldBuilder" prompt-interpreter is therefore just *another model id on
that endpoint*, not a new integration.

---

## 1. Where we actually are (verified 2026-10-15)

| Consumer | Today | Evidence |
|---|---|---|
| hyworld2 **worldgen stages 1–2** | ✅ Already remote. `OpenAI(api_key="EMPTY", base_url="http://{addr}:{port}/v1")`, `chat.completions.create(model, messages, max_tokens=1024, temperature=0.0, seed=1024)` | `build-context/hyworld2/worldgen/traj_generate.py:206,270,686`; `src/vlm_utils.py:150,185,230,313` |
| hyworld2 **pano** | No LLM. Qwen-Image-Edit's own text encoder, in-container | `hyworld2/panogen/pipeline_with_qwen_image.py:101` |
| hyworld2 **recon** | No LLM at all (no text conditioning) | `hyworld2/worldrecon/**` has zero tokenizer/LLM references |
| Lyra-2 **dynamic captioner** | ❌ **In-process.** `QwenCaptioner` loads `Qwen3-VL-4B-Instruct` with `AutoModelForImageTextToText`, keeps it on CPU and moves it to CUDA for every caption | `Lyra-2/gui/api/qwen_captioner.py:47-135`; called from `gui/api/lyra_persistent.py:164,394-397,525`; `ENV LYRA_GUI_QWEN_MODEL=/data/weights/qwen/Qwen3-VL-4B-Instruct` (`build-lyra2/docker/Dockerfile.service:38`) |
| both, text conditioning | UMT5-xxl encoder, in-container, unchanged by this plan | hyworld2 `worldstereo_wrapper.py:193,381` (Wan `tokenizer/`); Lyra `get_t5_emb.py:520` (`google/umt5-xxl`) |

**So there is exactly one divergence to remove: Lyra's bundled VLM.**

What that bundle costs: **8.3 GB** of weights on the weights drive
(`H:\AI_Models\lyra2\weights\qwen\`), a CPU-resident 4B model between captions, and a
CPU↔CUDA round-trip per caption (`on_device()` context manager) that fights the video
net for VRAM. The caption call sits inside `POST /request-inference`, which already
blocks for the whole chunk.

### 1.1 Foundry Local facts measured on this machine

- CLI/daemon **0.11.0**, `foundry status -o json` → `state:"ready"`,
  `webUrls:["http://127.0.0.1:49802"]` (auto-assigned port), ORT 1.30 / ORT-GenAI 2.1.
- `/v1/models` answers; **149 variant rows / 49 models**; 9 are `Multimodal`:
  `qwen3-vl-2b|4b|8b-instruct`, `qwen3.5-0.8b|2b|4b|9b`, `gemma-4-e2b-it`,
  `ministral-3-3b-instruct-2512`.
- **`qwen3-vl-8b-instruct` is already cached** (6130 MB) — and it is the default
  `llm_name` in `pipeline_spec.py:280`. One cached model can serve both world models.
- **No custom-model import.** `foundry model` has only `list / info / download / load /
  unload`. A fine-tuned adapter cannot be served by Foundry (0.11.0) — see §5.
- Served ids are **variant-qualified** (`qwen3-vl-8b-instruct-cuda-gpu:2`), not the bare
  alias. Whether the daemon also accepts the alias is **unverified** → gate G-1.
- Foundry binds `127.0.0.1` only, no bind-host option → the container needs the
  `netsh v4tov4` portproxy the app already creates (`PLAN.md` gotcha 50).

---

## 2. Goal and non-goals

**Goal:** one endpoint, one model id, one config location, engine-agnostic.

Principles:

1. **The endpoint is host-level, not per-backend.** It belongs beside the server
   address in Settings, not inside a world model's option table. (Matches
   `PLAN-MODEL-AGNOSTIC.md` §1: backend = data; the LLM is a shared service.)
2. **Vision is the requirement that picks the model.** Both consumers send images.
   A single vision-capable model (`qwen3-vl-*` family) covers both; do not run two.
3. **Engine swap must be a config edit, never a code edit.** Foundry → vLLM/SGLang
   changes only host:port + model id (+ bearer token, which Foundry does not need).
4. **Never depend on engine-specific extensions.** No `file://` image refs, no
   assuming `seed` is honoured, no `response_format`.
5. **Degrade, don't die.** A caption failure falls back to the user's own prompt text;
   it must not abort a generation.

**Non-goals:** no gateway/proxy service, no role registry, no per-role model routing,
no multi-model fallback, no headless server build in this plan (§5 only keeps the door
open). One endpoint, one model.

### 2.1 Why there is no gateway (decided 2026-10-15)

The requirement is: *run on one machine today; serve many users later by adding
hardware.* A gateway does not unlock that, because the abstraction that unlocks it
already exists — every hop is HTTP to a configurable address:

```
WinUI app ──http+SSE──> control service (container) ──subprocess──> world model
     │                        │
     └── host:port ───────────┴── host:port ──> LLM endpoint (/v1)
```

The app's service addresses are already settings rather than literals
(`WorldBuilder/Services/AppSettings.cs:15-16` → 8300, `:100-101` → 8400), and the LLM
is already a `host:port` string. **Scale-up is therefore "change the URLs", which needs
no new component.** The control service is already the single front door for the app;
a gateway would front a front door.

Co-locating a gateway with the world generator is technically fine (it is a small CPU
process and touches no VRAM). It is rejected on operational grounds: it adds a process
to start, a place for bugs to hide, and a second owner of "who talks to whom", while
providing nothing on a single machine that the control service does not already
provide.

What actually limits scaling, and what multi-user really needs:

| Limit | Evidence | Multi-user answer |
|---|---|---|
| Job queue and child-process handles are in-process, so `uvicorn workers=1` | `AGENTS.md` §6; `main.py:93` derives `queued` from `manager.jobs` | one container per concurrent GPU job + something that maps user → container |
| Loopback convenience | `resolve_llm_endpoint()` rewrites `localhost` → `172.22.32.1` | must be disabled when the LLM is genuinely remote (Phase 3) |
| SSE is hand-rolled `StreamingResponse` | `AGENTS.md` §8 | any future proxy must disable buffering and raise idle timeouts or progress freezes |
| No auth at all | `api_key="EMPTY"`; no token in `AppSettings` | TLS + tokens at a reverse proxy; Foundry needs neither |

Conclusion: hardware scale-up yields *N single-user machines*. Multi-user on one box is
a container-scheduling problem, not an LLM-routing problem. When it arrives, use
off-the-shelf parts (Caddy/nginx/Traefik for TLS + auth in front of the control
service; vLLM/SGLang for the LLM) — configuration, not a component authored here.
Nothing in Phases 0–3 has to be undone to add that later.

---

## 3. The contract

Both containers already speak, or will speak, exactly this:

```
POST http://{endpoint}/v1/chat/completions
{ "model": "...", "messages": [...], "max_tokens": N, "temperature": T }
```

with `messages` content parts limited to `{"type":"text"}` and
`{"type":"image_url","image_url":{"url":"data:image/{jpeg,png};base64,..."}}`.

The two call patterns that must keep working:

| Pattern | Used by | Needs |
|---|---|---|
| **Caption** — 2–5 images + long system prompt → 80–100 words of plain prose | Lyra dynamic captioner (`qwen_captioner.py` `CAPTION_SYSTEM_PROMPT`), hyworld2 `get_traj_caption` (`vlm_utils.py:149`), motion caption (`:230`) | vision, greedy-ish (`temperature` 0.0–0.1), `max_tokens` 100–1024, prose out |
| **Judge / select** — images or prior text → strict JSON | hyworld2 `env_cls` (`traj_generate.py:264`), `describe_splitted_images`, `wonder_selection` (`vlm_utils.py:94`) | vision, `temperature=0.0`, JSON object parseable from `choices[0].message.content` |

Config surface (three names, same meaning everywhere):

| Variable | Meaning | Default |
|---|---|---|
| `HYW2_LLM_ENDPOINT` / `LYRA_LLM_ENDPOINT` | `host:port` typed the way the **user** reaches it (`localhost:8000`); loopback is rewritten to the container gateway by the existing `resolve_llm_endpoint()` | `localhost:8000` |
| `*_LLM_MODEL` | model id the endpoint answers to | `qwen3-vl-8b-instruct` |
| `*_LLM_API_KEY` | bearer token; empty = `EMPTY` (Foundry needs none, vLLM/SGLang may) | empty |

---

## 4. Phases

### Phase 0 — Probe the endpoint (settles every unknown, no code changes)

Add `scripts/probe-llm.py <endpoint> [model]` (stdlib + `openai`, runs from WSL and
from inside a container) that reports, per model:

- **G-1 alias**: does `qwen3-vl-8b-instruct` resolve, or is
  `qwen3-vl-8b-instruct-cuda-gpu:2` required? (If required, the default `llm_name` is
  wrong and the app's served-models combo is the only thing saving it.)
- **G-2 vision**: one tiny base64 PNG → non-empty, image-aware answer.
- **G-3 JSON**: the `env_cls` prompt verbatim → output parses as expected.
- **G-4 params**: does it accept `seed` (hyworld2 sends it) and respect `max_tokens`?
  Rejecting an unknown kwarg would kill stage 1 on the first call.
- **G-5 latency**: ms for a 2-image caption — feeds the timeout in Phase 2.
- **G-6 reachability from the container** through `172.22.32.1` with the portproxy in
  place (already proven for the daemon; re-prove for the chosen model).

*Exit:* every gate has a recorded answer in this file.

### Phase 1 — One config location

- Promote the LLM from a worldgen run-option to **service config**: `settings.py` reads
  `HYW2_LLM_ENDPOINT/_MODEL/_API_KEY`; `pipeline_spec.backend_args()` falls back to it
  when the request omits `llm_endpoint`. The Run-page VLM card stays (it is a
  per-run override + the reachability probe), but its initial value comes from the
  service, so Lyra and HY-World cannot drift apart.
- Mirror it in `lyra2:service`: `LYRA_LLM_ENDPOINT/_MODEL/_API_KEY` in
  `build-lyra2/docker/Dockerfile.service`, surfaced through `GET /capabilities`.
- `GET /vlm/probe` gains `model` + `vision: true|false|unknown` so the app can warn
  "this model can't see" before a job dies 20 s in.
- App: the endpoint/model fields move to one **Language model** card in Settings
  (shared), the Run-page card becomes the override view.

*Exit:* setting one value in Settings drives both containers; `test_logic.py` green.

### Phase 2 — Lyra calls the shared endpoint (the real work)

Replace the bundled VLM with an HTTP client, behind a flag so rollback is one env var:

```
LYRA_CAPTIONER = remote | local | none      # default remote
```

- New `gui/api/remote_captioner.py`: same `caption(images, region_hint) -> str`
  signature as `QwenCaptioner`, so `lyra_persistent.py` is unchanged except for which
  class it constructs. Reuses `CAPTION_SYSTEM_PROMPT` verbatim; converts the same
  numpy/tensor images to base64 JPEG data URLs; sends `temperature=0.0`,
  `max_tokens=160`.
- `local` keeps `QwenCaptioner` exactly as-is (the escape hatch, and the reference
  implementation for output parity).
- `none` skips captioning and uses the user's `region_hint` / static prompt directly.
- **Timeout + fallback (mandatory):** on timeout, non-200, or empty text, log a warning
  and fall back to `region_hint` or the previous caption. Never raise into the
  generate path — the request already blocks for the whole chunk.
- Drop `openai` in favour of `httpx`/`urllib` if it keeps the Lyra wheelhouse smaller;
  the request is one POST.
- After Phase 2 passes: `LYRA_GUI_QWEN_MODEL` and the 8.3 GB
  `weights/qwen/` tree become removable from the image and the weights drive, and the
  `on_device()` VRAM ping-pong disappears.

*Exit:* Lyra E2E (seed → generate → export → PLY) with `LYRA_CAPTIONER=remote`, caption
text comparable to the local model's; then with `weights/qwen/` unmounted.

### Phase 3 — Engine-agnostic hardening (the headless/server door)

Cheap now, expensive later:

- **Bearer tokens everywhere.** Foundry ignores auth; vLLM/SGLang behind a reverse
  proxy will not. `Authorization: Bearer` when `*_LLM_API_KEY` is set.
- **No Foundry-only assumptions.** Anything Foundry-specific (portproxy, auto-port,
  alias-vs-variant ids) must live in the app's host-side helpers, not in either
  container's request path.
- **TLS + remote host** accepted by `resolve_llm_endpoint()` (today it strips the
  scheme and demands `host:port`; a server endpoint may be `https://host/v1`).
- **Loopback rewrite becomes opt-in.** Rewriting `localhost` → the container gateway is
  correct only when the LLM runs on the same physical machine; a remote endpoint must
  pass through untouched. Gate it on "the endpoint names this machine", not on the
  hostname list alone.
- **Every address is config, none are implied.** Audit for `127.0.0.1`/`localhost`
  defaults in `AppSettings`, `settings.py`, and `build-lyra2` so that a deployment is
  describable as one set of URLs: control service, Lyra service, LLM, weights root,
  output root. That set *is* the deployment; moving it is the scale-up.
- Document the swap in one place: *to move to vLLM/SGLang, run an OpenAI-compatible
  server with a vision model and change `*_LLM_ENDPOINT` + `*_LLM_MODEL`.* Nothing else.

*Exit:* the same job runs against Foundry Local and against any other
OpenAI-compatible endpoint with config changes only.

### Phase 4 — The fine-tuned WorldBuilder interpreter (future, unblocked by 0–3)

What it is: a model that turns loose user intent ("make it a foggy harbour at dusk,
walk toward the lighthouse") into the structured request each backend wants — a Lyra
`region_hint`, a worldgen prompt/stage/trajectory choice. It sits **in front of** the
world models, not between their internals.

Hard constraint found this session: **Foundry Local 0.11.0 cannot host custom or
fine-tuned weights** (`foundry model` has no import verb; it serves its own catalog).
So the fine-tune runs on the engine the server will use anyway — vLLM/SGLang, or
llama.cpp for a GGUF export. This is why §2's engine-agnostic rule matters: the
fine-tune arrives as a new `*_LLM_MODEL` value, and the app needs no change.

Training path (tooling already on this machine — Unsloth Studio, verified reachable):
dataset = the corpus the app already produces (user prompts + `region_hint`s from the
Lyra SPA, job logs, VLM captions and JSON verdicts from `GET /logs/{id}`), LoRA on a
Qwen base, `export_model(format="gguf")` for llama.cpp or merged for vLLM.
Before that, Phase 4a is simply **start logging** prompt→result pairs so a dataset
exists; that is a small, additive change and is worth doing regardless.

---

## 5. Risks / open questions

| # | Risk | Settled by |
|---|---|---|
| R1 | Foundry needs the variant-qualified id, so the default `llm_name` is wrong | G-1 |
| R2 | Foundry's shim rejects `seed` (hyworld2 sends it) → stage 1 dies on first call | G-4 |
| R3 | Foundry's vision quality differs from vLLM's Qwen3-VL (stage-1 E2E was verified against a *different* server, freetoken `Qwen3.8-27B-NVFP4`, never through Foundry — `PLAN.md` §8) | G-2/G-3 + a stage-1 run |
| R4 | Remote captioning adds latency to an already-blocking Lyra request | G-5, then the Phase 2 timeout |
| R5 | Caption quality regression vs the local 4B model | A/B the same seed with `LYRA_CAPTIONER=local` vs `remote` |
| R6 | VRAM: ~52 GB of the 96 GB GPU is normally held by the user's own inference server; Foundry's 6.1 GB model loads on top | `foundry status` + `nvidia-smi` before the first run |
| R7 | A fine-tuned model can never run "in Foundry", which may surprise a future reader | §4 Phase 4 states it; keep the wording in Settings UI honest |
| R8 | Loopback rewrite silently misroutes a *remote* LLM to the local gateway | Phase 3 audit; a probe that echoes back the address it dialed |
| R9 | A future reverse proxy buffers SSE and progress bars freeze | documented in §2.1; verify with one proxied run when a proxy appears |

---

## 6. Acceptance ladder

1. `scripts/probe-llm.py` reports all six gates against the live Foundry daemon.
2. hyworld2 stage 1 E2E through the service with the endpoint coming from **service
   config**, not a per-run option.
3. Lyra E2E with `LYRA_CAPTIONER=remote`, weights drive **without** `qwen/`.
4. Both containers point at a **non-Foundry** OpenAI endpoint (any vLLM/SGLang) and
   both E2E paths still pass — this is the proof the abstraction is real.
5. `python3 test_logic.py` and `python3 test_jobs_smoke.py` green throughout.

---

## 7. Verification notes for a reviewer

Claims worth checking independently, in order of how much they matter:

1. **Lyra is the only divergence.** Grep `Lyra-2` for `OpenAI|chat.completions`
   (expect none) and for `AutoModelForImageTextToText` (expect `qwen_captioner.py`);
   grep `build-context/hyworld2` for the same (expect `vlm_utils.py` +
   `traj_generate.py` only).
2. **Foundry cannot host a fine-tune.** `foundry model --help` → no import verb.
3. **`qwen3-vl-8b-instruct` is cached and multimodal.** `foundry model list -o json`.
4. **The caption path is in the blocking request.** `lyra_persistent.py:525` inside
   `request_inference`, before the chunk loop.
5. **UMT5 is untouched by this plan.** hyworld2 `worldstereo_wrapper.py:193` and Lyra
   `get_t5_emb.py:520` both resolve UMT5-xxl and neither is an LLM call.

Things this plan deliberately does **not** claim: that Foundry's shim supports
`response_format`/JSON mode (untested, and the plan avoids depending on it); that the
fine-tune will be a Qwen (base model is a Phase 4 decision); that any server-side
deployment exists yet.

---

## Suggestions from Junie model

### 1. Robustness & Fallback Strategy (Client Resilience)
- **Short Timeouts & Minimal Retries:** In Phase 2 (`remote_captioner.py`), captioning sits in the blocking inference loop (`POST /request-inference`). Enforce a strict timeout (e.g., 2.5–4.0 s) with at most 1 fast retry (100–200 ms backoff) before gracefully falling back to `region_hint`/static prompt.
- **Image Pre-processing & Downsampling:** Downsample image inputs (e.g., max dimension 448–512px, JPEG quality 80–85%) prior to base64 encoding. Vision models do not need full-resolution frames for dynamic captioning, and this drastically cuts JSON payload size, base64 encoding overhead, and network transfer time.
- **Circuit Breaker / Degraded State:** If $N$ consecutive caption requests fail or time out (e.g., daemon crash or cold unload), trip a temporary circuit breaker to bypass LLM calls for $M$ seconds, avoiding sequential blocking timeouts across frame chunks.

### 2. Unified Client & Schema Abstraction
- **Shared Client Utility / Schema:** Instead of maintaining distinct OpenAI/HTTP caller implementations across Lyra and HY-World, build a shared lightweight client helper (e.g., with `httpx`) standardizing headers (`Authorization: Bearer`), error handling, and payload schemas.
- **Robust JSON Extraction:** For HY-World judge/selection calls (`env_cls`, `wonder_selection`), implement a resilient markdown-stripping JSON extractor (` ```json ... ``` ` and regex extraction) to ensure compatibility across engines that may or may not support `response_format: {"type": "json_object"}`.

### 3. Endpoint Configuration & Networking
- **Base URL Normalization:** Support full URL base strings (`http://localhost:8000/v1` or `https://remote-server/v1`) in `*_LLM_ENDPOINT` alongside `host:port` to handle custom reverse proxy subpaths cleanly.
- **Persistent HTTP Sessions:** Use persistent connection pooling / keep-alive (`httpx.Client`) rather than instantiating connections per request to eliminate TCP handshake and portproxy overhead.

### 4. Concurrency & GPU Contention
- **VRAM Profiles:** Foundry's `qwen3-vl-8b-instruct` (~6.1 GB) coexists with world models on the GPU. Allow fallback configuration options for smaller variants (`qwen3-vl-4b` or `qwen3-vl-2b`) if peak VRAM usage during concurrent diffusion and captioning is constrained.
- **Non-blocking GUI Probes:** Ensure capability probing (`GET /vlm/probe`) in the UI executes asynchronously with tight timeouts so slow or cold LLM starts never block container readiness or GUI rendering.
