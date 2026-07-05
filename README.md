# ai-image

ComfyUI-based local image app for FLUX.2 Klein 4B on consumer NVIDIA GPUs.

## What this repo contains

- `comfyui_app/` - the active app, workflow builders, resolver, and UI
- `Install.bat` - first-time setup
- `Install-SageAttention.bat` - install the matching SageAttention wheel
- `Install-TeaCache.bat` - install the optional TeaCache custom node
- `Launch.bat` - start ComfyUI and the Gradio UI
- `Cleanup-Unused-Models.bat` - headless model cleanup
- `Update.bat` - in-place update without rebuilding the Python environment

The old Hugging Face diffusers app has been removed from the repository.

## Setup

Run `Install.bat` once on a new machine or when you want a fresh environment.

What it does:

1. Creates the local virtual environment if needed.
2. Installs the CUDA PyTorch stack.
3. Installs the ComfyUI helper dependencies.
4. Clones ComfyUI and the required custom nodes.
5. Resolves and downloads the model files into the ComfyUI model folders.

## Updating

Run `Update.bat` after code changes.

It updates in place:

1. `git pull`
2. `pip install -r requirements-comfyui.txt`
3. `python -m comfyui_app.installer`

`Update.bat` does **not** reinstall PyTorch or rebuild the virtual environment.

## Launch

Run `Launch.bat` and open the app at:

- http://127.0.0.1:7861

ComfyUI itself listens on `127.0.0.1:8188`.

## App tabs

- **Image Edit** - upload a depth source image, optionally add a separate reference image, and edit with FLUX.2 Klein.
- **Video to Frames** - extract frames from a video, then optionally batch-edit the extracted frames.
- **Batch Folder** - batch-edit a folder of images.
- **Text-to-Image** - generate from scratch.
- **Upscale** - upscale a single image or a folder of images with either RTX VSR or Real-ESRGAN.
- **Video Upscale** - upscale a whole video and write an H.264/AAC `.mp4` output.

Each tab has an output folder `Browse...` button, and the long-running tabs have a `Stop` button for cancellation.
The Text-to-Image and Image Edit tabs also have an optional **Show live preview** toggle that streams ComfyUI websocket preview frames while a job is sampling.

## Model layout

The resolver writes into the standard ComfyUI folders:

- `ComfyUI/models/diffusion_models/`
- `ComfyUI/models/text_encoders/`
- `ComfyUI/models/vae/`
- `ComfyUI/models/upscale_models/`
- `ComfyUI/models/loras/`

## Manage Models

Use the **Manage Models** tab to review installed model files, see their sizes, and delete ones you no longer need to free disk space. It also includes a cleanup preview for unused / duplicate models before you confirm deletion.

- It shows the file category, filename, and size.
- Deleted files are removed from disk and from the resolved manifest.
- If you need a deleted model again later, the app will re-download it on demand when you pick that engine or feature path again.
- The cleanup flow is also available headlessly via `Cleanup-Unused-Models.bat` from the repository root.

## Upscaling

### RTX Video Super Resolution

The NVIDIA RTX VSR path uses the `RTXVideoSuperResolution` node from the Comfy-Org RTX node package. It requires:

- an NVIDIA RTX GPU
- the `nvidia-vfx` runtime package
- the `Comfy-Org/Nvidia_RTX_Nodes_ComfyUI` custom node

The installer tries to add that support automatically, but it is best-effort. If the RTX runtime is unavailable, the app falls back to Real-ESRGAN x2+.

### Real-ESRGAN fallback

The fallback upscaler uses the core ComfyUI `UpscaleModelLoader` + `ImageUpscaleWithModel` path and the already-resolved `RealESRGAN_x2plus.pth` checkpoint in `ComfyUI/models/upscale_models/`.

## Engine choices

- **INT8** is the default engine in the UI and install flow.
- The app is now INT8-only for the main diffusion path.
- **GGUF Q4/Q5** remains available on the lower-VRAM tier path.

### Video output

Video upscaling reassembles frames with `imageio-ffmpeg` and preserves the original audio track when present:

- input: `.mp4` or `.mov`
- output: `.mp4` with H.264 video
- audio: copied back in when the source has an audio stream
- no audio stream: the output stays video-only

## Browse buttons and Stop

The UI uses native folder pickers for folder fields. Clicking `Browse...` fills the current textbox value without blocking manual edits.

Long-running tabs include a `Stop` button. It:

1. sets the in-app cancel flag
2. calls ComfyUI's `/interrupt` endpoint
3. cancels the frontend event when Gradio supports it

## Batch output behavior

Batch jobs now write into a timestamped per-run subfolder under the chosen output folder, and the Batch / Upscale folder tabs stream results into a live gallery as each image completes.

## RTX 3070 / 8 GB default plan

| Area | Default | Why |
| --- | --- | --- |
| Diffusion | FLUX.2 Klein 4B INT8 | Native INT8 tensor cores on Ampere are faster than the older baseline path, with the same VRAM class. |
| Text encoder | Qwen 3 4B fp4 | Lower VRAM than fp16/bf16. |
| VAE | FLUX.2 small decoder | Best default fit for the app's edit flow. |
| Decode | Tiled by default on low-memory tiers | Helps keep 8 GB cards from spiking. |
| Launch flags | `--fast fp16_accumulation --reserve-vram 0.8 --fast-disk` | Ampere-friendly speed and safer Windows headroom. |

Why this stack:

- The RTX 3070 is Ampere, so INT8 runs on native tensor cores.
- Expect roughly a 25-35% steady-state speedup versus the older baseline path, with near-lossless quality for this model.
- The first run in a session is slower because kernels warm up; after that the INT8 path is the fastest default.
- The small decoder VAE keeps the edit pipeline compact.
- ComfyUI's offload model works well for the 8 GB target.

Sources:

- https://docs.comfy.org/
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/black-forest-labs/FLUX.2-small-decoder

## Optional speedups

- **SageAttention 2** is optional and can be installed with `Install-SageAttention.bat`. `Install.bat --with-experimental-speedups` also installs the matching wheel automatically.
- **TeaCache** is optional and experimental. Install it with `Install-TeaCache.bat`, or let `Install.bat --with-experimental-speedups` install the custom node for you.
- **Nunchaku INT4** is experimental but can be much faster. Use the `tonera/FLUX.2-klein-4B-Nunchaku` INT4 checkpoint and the experimental installer path if you want to try it.
- **torch.compile** is available in the UI. It requires Triton on Windows, which is installed via the experimental speedups path (`--with-experimental-speedups`). The gain on Ampere is limited, but it can help after warmup. The first run is slower and resolution changes recompile.

## MrFlow staged sampling (experimental)

MrFlow is a staged-sampling acceleration path inspired by https://github.com/Xingyu-Zheng/MrFlow and https://arxiv.org/abs/2607.01642.

In this app it keeps the same FLUX.2 Klein loaders and defaults, but runs:

1. a low-resolution first pass,
2. VAE decode,
3. 2x upscaling with a Real-ESRGAN model,
4. VAE re-encode,
5. a short high-resolution refinement pass.

Defaults:

- stage 1: 4 steps
- refine: 1 step
- refine denoise: 0.25
- low-res size: 512x512 for a 1024x1024 target

Notes:

- It works with the default FLUX.2 small-decoder VAE.
- It is experimental and can drift more on edits than a direct full-resolution edit.
- It is off by default.

The app auto-downloads `RealESRGAN_x2plus.pth` into `ComfyUI/models/upscale_models/` during the normal model refresh.

## Pose/Shape lock (depth reference) - optional

The depth Pose/Shape lock path uses the **RefControl** depth LoRA for FLUX.2 Klein 4B.

What it does:

- runs the input image through `DepthAnythingV2Preprocessor`
- uses that depth map to lock pose / shape
- optionally uses a second reference image for identity/style
- keeps the normal fast 4-step distilled paths unchanged when the checkbox is off

Requirements:

- base model: `unsloth/FLUX.2-klein-base-4B-GGUF` (`flux-2-klein-base-4b-Q8_0.gguf`)
- depth LoRA: `thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora`
- custom node: `Fannovel16/comfyui_controlnet_aux`

Install it with:

```bat
python -m comfyui_app.installer --with-depth-control
```

Notes:

- Trigger word: `refcontrol`
- This path uses the undistilled base model, so it is slower than the normal 4-step edit path.
- The 8-bit Q8_0 GGUF base is the default here and loads through ComfyUI-GGUF; on Ampere it should still run efficiently while avoiding the raw safetensors/ModelOpt path.
- Expect roughly a ~20 step workflow and about a 5x slowdown versus the distilled edit path.
- For FLUX.2 Klein 4B, only the **depth** lock mode is available here.

## Consistency LoRA (optional)

The optional consistency LoRA improves edit and generation fidelity by keeping subjects and details more consistent across runs.

Install it with:

```bat
python -m comfyui_app.installer --with-consistency-lora
```

Source:

- HF repo: `lrzjason/Consistance_Edit_Lora`
- file: `f2k_4B_consist_20260314.safetensors`
- license: Apache-2.0

Strength guidance:

- higher values push for more consistency
- lower the strength if edits stop applying cleanly

It is available in:

- Image Edit
- Text-to-Image
- Batch Folder

## Dependencies

The ComfyUI helper requirements include `imageio-ffmpeg` so the app can reassemble videos without relying on a system ffmpeg install.

## Working with the resolver

The app resolves models from Hugging Face and skips files that are already present locally. That makes normal updates fast after the first run.

## ComfyUI references

- https://github.com/comfyanonymous/ComfyUI
- https://docs.comfy.org/development/comfyui-server/comms_overview
- https://docs.comfy.org/development/core-concepts/models
