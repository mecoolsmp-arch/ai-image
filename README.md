# gradio-ai-v2

## ComfyUI local image app (FLUX.2 [klein] 4B, RTX 3070 optimized)

This repo now includes an additive, ComfyUI-based local image app under
`comfyui_app/` that replaces the Hugging Face **diffusers** WebUI for the
FLUX.2 [klein] 4B workflow. ComfyUI is faster and much more VRAM-efficient for
this model on 8 GB cards, so the full functionality (image editing,
video-to-frames, batch folder processing, configurable output) was migrated on
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
│   │   └── vae/                     # flux2-vae.safetensors
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

The resolver also **probes for a tiny/"small" VAE decoder** (e.g. a future
`taef2`-style tiny autoencoder) and prefers it automatically if one is ever
published. See the VAE note below.

### RTX 3070 (8 GB) optimization — what was chosen and why

Detected VRAM selects a tier (`comfyui_app/vram.py`). On an RTX 3070 (8 GB) the
resolver lands in the **7–9 GB** tier:

| Setting | Choice for 8 GB | Why |
| --- | --- | --- |
| Diffusion model | `flux-2-klein-4b-fp8.safetensors` (~4.0 GB) | fp8 halves the weight footprint vs bf16 (~7.75 GB) and pairs with ComfyUI's `--fast` fp8 matmul path. The distilled Klein 4B needs ~8.4 GB total; fp8 + ComfyUI's smart offload fits 8 GB. |
| Diffusion fallback | GGUF `Q4_K_M` (~2.6 GB), then `Q3`/`Q2` | If fp8 OOMs, the app automatically retries with a lighter GGUF quant via the `UnetLoaderGGUF` node (ComfyUI-GGUF). Q4_K_M keeps most quality at ~2.6 GB. |
| Text encoder | `qwen_3_4b_fp4_flux2.safetensors` (~3.85 GB) | The full Qwen3-4B encoder is ~8 GB and won't co-reside; the fp4 build is half the size. ComfyUI runs the encoder, frees it, then loads the diffusion model, so peak VRAM stays manageable. |
| VAE | `flux2-vae.safetensors` (~0.34 GB) | This is FLUX.2's compact VAE (see note). |
| Decode | `VAEDecodeTiled` | Tiled decode caps peak VRAM during decode on 8 GB, preventing end-of-run OOM at higher resolutions. |
| Sampler / scheduler | `euler` + `Flux2Scheduler` | Matches the official distilled Klein template. |
| Steps | `4` | Distilled Klein is a 4-step model; more steps waste time without quality gains. |
| Guidance (CFG) | `1.0` | Distilled Klein uses CFG 1. See the negative-prompt note below. |
| Launch flags | `--fast`, `--use-sage-attention` (if installed) | `--fast` enables the fp8 fast path; SageAttention gives a 2–5× attention speedup. `--lowvram` is added only on the sub-7 GB tiers. |

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

- **"Small VAE decoder":** FLUX.2 does not (yet) ship a separate tiny/distilled
  decoder (there is no `taef2` equivalent). `flux2-vae.safetensors` **is** the
  compact VAE (only ~340 MB), and on 8 GB the app uses `VAEDecodeTiled` for the
  fastest, lowest-peak-VRAM decode. The resolver auto-prefers a tiny decoder if
  one is ever released.
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
