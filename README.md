# gradio-ai-v2

## ComfyUI local image app

This repo now includes an additive, ComfyUI-based local image app under
`comfyui_app/` that replaces the Hugging Face **diffusers** WebUI for the
FLUX.2 [klein] 4B workflow. ComfyUI is faster and more VRAM-efficient on
8 GB cards, and the new app keeps the existing image-edit, video-frame, batch,
and text-to-image workflows.

### Model layout

- `models/diffusion_models/`
- `models/text_encoders/`
- `models/vae/`
  - Default: `black-forest-labs/FLUX.2-small-decoder` →
    `full_encoder_small_decoder.safetensors`
  - Fallback: `Comfy-Org/flux2-klein-4B` → `flux2-vae.safetensors`

The small decoder is the preferred VAE for this app. It is a distilled,
drop-in decoder that is about 1.4× faster to decode and uses about 1.4× less
VRAM than the standard `flux2-vae` decoder, with minimal quality difference.
For image edit and batch workflows we still use the standard VAE encode path;
the single-file `full_encoder_small_decoder.safetensors` is used because it
contains the normal encoder plus the faster decoder.

### RTX 3070 / 8 GB default plan

- Diffusion: `flux-2-klein-4b-fp8.safetensors`
- Text encoder: `qwen_3_4b_fp4_flux2.safetensors`
- VAE: `full_encoder_small_decoder.safetensors`
- Decode: `VAEDecodeTiled`

### Repos checked

- `black-forest-labs/FLUX.2-klein-4b-fp8`
- `black-forest-labs/FLUX.2-small-decoder`
- `Comfy-Org/flux2-klein-4B`
- `unsloth/FLUX.2-klein-4B-GGUF`

### Sources

- https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8
- https://huggingface.co/black-forest-labs/FLUX.2-small-decoder
- https://huggingface.co/Comfy-Org/flux2-klein-4B
- https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF
top of ComfyUI's HTTP API behind a simple, non-technical Gradio front end — no
node graph required.

The original diffusers Gradio app (`app.py`, `src/`, `Install.bat`,
`Launch.bat`) is left **unchanged** so you can fall back to it at any time.

### Quick start (Windows)

1. `Install-ComfyUI.bat` — creates the venv, installs the CUDA PyTorch stack,
   clones ComfyUI + the `ComfyUI-GGUF` and `ComfyUI-Manager` custom nodes,
   prompts once for a **Hugging Face token**, auto-detects your VRAM, and
   downloads the correct FLUX.2 Klein 4B model files into the right folders.
2. `Launch-ComfyUI.bat` — starts the ComfyUI server (with the launch flags
   chosen for your GPU) and opens the simplified UI at http://127.0.0.1:7861.

The Hugging Face token is saved to `.env` (which is git-ignored). Some FLUX.2
repos are gated — accept the license on the model page (linked in error
messages) before running setup. Re-running setup with
`python -m comfyui_app.installer --refresh-models` re-checks the repos and pulls
the current recommended files.

### What you get in the UI

- **Image Edit** – upload an image + prompt/negative prompt; text-guided edit.
- **Video to Frames** – upload a video, extract every Nth frame, then run the
  same edit pipeline over every extracted frame.
- **Batch Folder** – point at an input folder; one shared prompt + negative
  prompt is applied to every image in it.
- **Text-to-Image** – native FLUX.2 Klein text-to-image (bonus).
- Every tab has an **Output folder** field so results save where you choose
  (default: `output/`).

### Folder locations

```
repo/
├── ComfyUI/                         # cloned by the installer
│   ├── models/
│   │   ├── diffusion_models/        # flux-2-klein-4b-fp8.safetensors (or GGUF)
│   │   ├── text_encoders/           # qwen_3_4b_fp4_flux2.safetensors (or full)
│   │   └── vae/                     # full_encoder_small_decoder.safetensors (or flux2-vae)
│   └── custom_nodes/                # ComfyUI-GGUF, ComfyUI-Manager
├── comfyui_app/
│   ├── workflows/                   # flux2_klein_edit.json, flux2_klein_t2i.json
│   └── resolved_models.json         # manifest of what was downloaded (repeatable)
├── output/                          # default output directory
└── .env                            # HF_TOKEN (git-ignored)
```

### Model auto-install ("model resolver")

`comfyui_app/model_resolver.py` is a repeatable resolver rather than a set of
hard-coded links. For each component (diffusion model, text encoder, VAE) it
queries the Hugging Face API live
(`/api/models/<repo>/tree/main?recursive=1`) and picks the current file that
matches a stable filename pattern, so newer Klein 4B distilled checkpoints are
picked up automatically when you re-run setup. Repos checked:

- `black-forest-labs/FLUX.2-klein-4b-fp8` – fp8 distilled diffusion model
- `unsloth/FLUX.2-klein-4B-GGUF` – GGUF quantized fallbacks (Q4_K_M / Q3 / Q2)
- `Comfy-Org/flux2-klein-4B` – fp4 & full text encoders, VAE
- `black-forest-labs/FLUX.2-klein-4B` – reference (upstream distilled repo)

Files are downloaded with `huggingface_hub` (authenticated with your token),
flattened into `ComfyUI/models/{diffusion_models,text_encoders,vae}`, and
recorded in `resolved_models.json` (repo, remote path, size, sha) so unchanged
files are skipped on re-run. Gated/expired-token/unaccepted-license failures are
reported in plain language with the exact model page to visit.

For the VAE, the resolver **defaults to BFL's distilled small decoder**
(`black-forest-labs/FLUX.2-small-decoder`) and falls back to
`Comfy-Org/flux2-klein-4B`'s `flux2-vae.safetensors` only if that lookup fails.
See the VAE note below.

### RTX 3070 (8 GB) optimization — what was chosen and why

Detected VRAM selects a tier (`comfyui_app/vram.py`). On an RTX 3070 (8 GB) the
resolver lands in the **7–9 GB** tier:

| Setting | Choice for 8 GB | Why |
| --- | --- | --- |
| Diffusion model | `flux-2-klein-4b-fp8.safetensors` (~4.0 GB) | fp8 halves the weight footprint vs bf16 (~7.75 GB) so Klein fits in 8 GB with ComfyUI's smart offload. Note: the RTX 3070 (Ampere) has **no fp8 tensor cores** — fp8 is a VRAM win here, not a compute speedup (fp8 matmul acceleration starts at Ada/40-series). Because offloading is the main slowdown on 8 GB, a GGUF Q5/Q4 that stays fully resident can be *faster* end-to-end; see the GGUF fallback row. |
| Diffusion fallback | GGUF `Q4_K_M` (~2.6 GB), then `Q3`/`Q2` | If fp8 OOMs, the app automatically retries with a lighter GGUF quant via the `UnetLoaderGGUF` node (ComfyUI-GGUF). Q4_K_M keeps most quality at ~2.6 GB. |
| Text encoder | `qwen_3_4b_fp4_flux2.safetensors` (~3.85 GB) | The full Qwen3-4B encoder is ~8 GB and won't co-reside; the fp4 build is half the size. ComfyUI runs the encoder, frees it, then loads the diffusion model, so peak VRAM stays manageable. |
| VAE | `full_encoder_small_decoder.safetensors` (~0.25 GB) | BFL's distilled small decoder: ~1.4× faster decode and ~1.4× less VRAM than the standard `flux2-vae`, minimal quality loss. Full standard encoder is retained for the edit/video/batch encode path. `flux2-vae.safetensors` is the automatic fallback. |
| Decode | `VAEDecodeTiled` | Tiled decode caps peak VRAM during decode on 8 GB, preventing end-of-run OOM at higher resolutions. |
| Sampler / scheduler | `euler` + `Flux2Scheduler` | Matches the official distilled Klein template. |
| Steps | `4` | Distilled Klein is a 4-step model; more steps waste time without quality gains. |
| Guidance (CFG) | `1.0` | Distilled Klein uses CFG 1. See the negative-prompt note below. |
| Launch flags | `--fast fp16_accumulation --reserve-vram 0.8 --fast-disk`, `--use-sage-attention` (if installed) | `--fast` is scoped to `fp16_accumulation` (the feature that actually speeds up Ampere; bare `--fast` also enables `fp8_matrix_mult`, a no-op on the 3070 that can degrade quality). `--reserve-vram 0.8` leaves headroom for the Windows display to avoid OOM/offload stalls. `--fast-disk` speeds the unavoidable offload on NVMe SSDs (drop it if models live on an HDD). SageAttention gives a 2–5× INT8 attention speedup on Ampere. `--lowvram` is added only on the sub-7 GB tiers. |

Sources / references:

- ComfyUI Flux.2 Klein 4B guide (node structure, model files, storage layout):
  https://docs.comfy.org/tutorials/flux/flux-2-klein
- BFL model card / VRAM & latency figures (distilled ~1.2 s, ~8.4 GB on a 5090):
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- fp8 distilled weights: https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8
- GGUF quantized fallbacks: https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF
- Text encoder / VAE (Comfy-Org repack): https://huggingface.co/Comfy-Org/flux2-klein-4B
- SageAttention (Windows wheels): https://github.com/woct0rdho/SageAttention

### Notes / limitations

- **Small VAE decoder:** the app defaults to BFL's distilled
  `black-forest-labs/FLUX.2-small-decoder`
  (`full_encoder_small_decoder.safetensors`) — ~1.4× faster decode and ~1.4×
  less VRAM than the standard `flux2-vae` decoder, with minimal quality loss —
  combined with `VAEDecodeTiled` for the lowest peak VRAM on 8 GB.
  `flux2-vae.safetensors` is the automatic fallback if the small-decoder repo
  can't be resolved.
- **fp8 on the RTX 3070:** fp8 saves VRAM but does **not** accelerate compute on
  Ampere (no fp8 tensor cores). On 8 GB the dominant cost is weight offloading,
  so if you want more speed, try the GGUF path (Q5_K_M/Q4_K_M) which can keep the
  model fully resident and avoid offload.
- **Negative prompt on the distilled model:** the distilled Klein workflow runs
  at CFG 1.0, where the negative branch is mathematically ignored. The negative
  prompt field is still wired end-to-end (and shared across batch) so it takes
  effect if you raise Guidance above 1.0 — but the distilled model is not tuned
  for CFG > 1, so expect limited benefit. Leave Guidance at 1.0 for best speed.

### Changing models later

- Re-run `python -m comfyui_app.installer --refresh-models` to re-resolve and
  pull the current recommended files (picks up new releases automatically).
- To pin different files, edit `MODEL_REGISTRY` / the VRAM tiers in
  `comfyui_app/model_resolver.py` and `comfyui_app/vram.py` (repo + filename
  regex per component), then refresh.
- Delete `comfyui_app/resolved_models.json` (and the copy in `ComfyUI/`) to force
  a clean re-download.
