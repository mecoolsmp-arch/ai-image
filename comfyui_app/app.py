from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import gradio as gr
except Exception:  # pragma: no cover - optional dependency
    gr = None  # type: ignore[assignment]

from comfyui_app.batch import process_folder
from comfyui_app.comfy_client import ComfyClient
from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT, DEFAULT_OUTPUT_DIR
from comfyui_app.generation import GenerationResult, run_edit, run_t2i
from comfyui_app.model_resolver import ModelResolverError, load_resolved_manifest
from comfyui_app.vram import detect_vram, select_tier
from comfyui_app.video_frames import extract_frames

logger = logging.getLogger(__name__)

ENGINE_CHOICES = [
    ("fp8 (default)", "default"),
    ("Nunchaku INT4 (experimental — faster, needs extra install)", "nunchaku_int4"),
]


def _client() -> ComfyClient:
    return ComfyClient(COMFYUI_HOST, COMFYUI_PORT)


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, ModelResolverError):
        return exc.message
    text = str(exc).strip()
    first_line = text.splitlines()[0] if text else "Unknown error"
    return f"Something went wrong: {first_line}"


def _status_markdown() -> str:
    vram_gb, device_name, cuda_available = detect_vram()
    tier = select_tier(vram_gb)
    manifest = load_resolved_manifest()
    model_summary = "No model files have been prepared yet."
    if isinstance(manifest, dict):
        models = manifest.get("models")
        if isinstance(models, dict):
            diffusion = models.get("diffusion", {})
            text_encoder = models.get("text_encoder", {})
            vae = models.get("vae", {})
            model_summary = (
                f"- Diffusion: {diffusion.get('local_filename', 'unknown')}\n"
                f"- Text encoder: {text_encoder.get('local_filename', 'unknown')}\n"
                f"- Decoder: {vae.get('local_filename', 'unknown')}"
            )
    server_ready = False
    try:
        server_ready = _client().is_server_up()
    except Exception:
        server_ready = False
    gpu_line = "No NVIDIA GPU was detected." if not cuda_available else f"{device_name} with about {vram_gb:.1f} GB of VRAM."
    server_line = "ComfyUI is ready." if server_ready else "ComfyUI is not running yet."
    return (
        "### Setup status\n"
        f"- GPU: {gpu_line}\n"
        f"- Plan: {tier.label} — diffusion {tier.diffusion}, text encoder {tier.text_encoder}, "
        f"tiled decoder {'on' if tier.use_tiled_decode else 'off'}\n"
        f"- Server: {server_line}\n"
        f"\n{model_summary}"
    )


def refresh_status() -> str:
    return _status_markdown()


def _single_edit(
    input_image_path: str,
    prompt: str,
    negative: str,
    output_dir: str,
    steps: int,
    cfg: float,
    seed: int,
    megapixels: float,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> tuple[str | None, str]:
    try:
        result = run_edit(
            input_image_path=input_image_path,
            prompt=prompt,
            negative=negative,
            output_dir=output_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )
        return str(result.image_path), result.status
    except Exception as exc:
        return None, _friendly_error(exc)


def _extract_video(video_path: str, output_dir: str, every_n: int, max_frames: int) -> tuple[str | None, str]:
    if not video_path:
        return None, "Upload a video first."
    try:
        source = Path(video_path)
        frame_dir = Path(output_dir) / f"{source.stem}_frames"
        frames = extract_frames(source, frame_dir, every_n=max(1, int(every_n)), max_frames=max_frames or None)
        return str(frame_dir), f"Saved {len(frames)} frames to {frame_dir}."
    except Exception as exc:
        return None, _friendly_error(exc)


def _edit_frames(
    frame_dir: str,
    prompt: str,
    negative: str,
    output_dir: str,
    steps: int,
    cfg: float,
    seed: int,
    megapixels: float,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> str:
    if not frame_dir:
        return "Extract frames first."

    def _runner(image_path: Path, prompt_text: str, negative_text: str, target_dir: Path) -> GenerationResult:
        return run_edit(
            input_image_path=image_path,
            prompt=prompt_text,
            negative=negative_text,
            output_dir=target_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )

    try:
        summary = process_folder(frame_dir, output_dir, prompt, negative, _runner)
        message = summary.get("message")
        if isinstance(message, str):
            return message
        failures = summary["failures"]
        if failures:
            return f"Edited {summary['count']} frames, with {len(failures)} problems."
        return f"Edited {summary['count']} frames."
    except Exception as exc:
        return _friendly_error(exc)


def _process_batch(
    input_dir: str,
    prompt: str,
    negative: str,
    output_dir: str,
    steps: int,
    cfg: float,
    seed: int,
    megapixels: float,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> str:
    def _runner(image_path: Path, prompt_text: str, negative_text: str, target_dir: Path) -> GenerationResult:
        return run_edit(
            input_image_path=image_path,
            prompt=prompt_text,
            negative=negative_text,
            output_dir=target_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )

    try:
        summary = process_folder(input_dir, output_dir, prompt, negative, _runner)
        message = summary.get("message")
        if isinstance(message, str):
            return message
        failures = summary["failures"]
        if failures:
            return f"Processed {summary['count']} files, with {len(failures)} problems."
        return f"Processed {summary['count']} files."
    except Exception as exc:
        return _friendly_error(exc)


def _generate_t2i(
    prompt: str,
    negative: str,
    output_dir: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> tuple[str | None, str]:
    try:
        result = run_t2i(
            prompt=prompt,
            negative=negative,
            output_dir=output_dir,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )
        return str(result.image_path), result.status
    except Exception as exc:
        return None, _friendly_error(exc)


def build_app() -> "gr.Blocks":
    if gr is None:
        raise RuntimeError("Gradio is not installed.")

    with gr.Blocks(title="ComfyUI Local Image App") as demo:
        gr.Markdown("# ComfyUI Local Image App")
        status_box = gr.Markdown(_status_markdown())
        refresh_button = gr.Button("Refresh status")
        refresh_button.click(fn=refresh_status, outputs=status_box)

        with gr.Tab("Image Edit"):
            with gr.Row():
                with gr.Column():
                    edit_image = gr.Image(label="Image", type="filepath")
                    edit_prompt = gr.Textbox(label="Prompt", lines=4)
                    edit_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    edit_output = gr.Textbox(label="Output folder", value=str(DEFAULT_OUTPUT_DIR))
                    edit_steps = gr.Number(label="Steps", value=4, precision=0)
                    edit_cfg = gr.Number(label="Guidance", value=1.0)
                    edit_megapixels = gr.Number(label="Megapixels", value=1.0)
                    edit_seed = gr.Number(label="Seed", value=0, precision=0)
                    edit_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    edit_compile = gr.Checkbox(
                        label="torch.compile (faster after warmup; slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    edit_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental — faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_button = gr.Button("Generate")
                    edit_result = gr.Image(label="Result")
                    edit_status = gr.Textbox(label="Status")
            edit_button.click(
                fn=_single_edit,
                inputs=[
                    edit_image,
                    edit_prompt,
                    edit_negative,
                    edit_output,
                    edit_steps,
                    edit_cfg,
                    edit_seed,
                    edit_megapixels,
                    edit_engine,
                    edit_compile,
                    edit_mrflow,
                ],
                outputs=[edit_result, edit_status],
            )

        with gr.Tab("Video to Frames"):
            frame_state = gr.State("")
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video", type="filepath")
                    every_n = gr.Number(label="Every Nth frame", value=1, precision=0)
                    max_frames = gr.Number(label="Max frames", value=0, precision=0)
                    video_output = gr.Textbox(label="Output folder", value=str(DEFAULT_OUTPUT_DIR))
                    extract_button = gr.Button("Extract frames")
                    frame_status = gr.Textbox(label="Status")
                with gr.Column():
                    video_prompt = gr.Textbox(label="Prompt", lines=4)
                    video_negative = gr.Textbox(label="Negative prompt", lines=3)
                    video_steps = gr.Number(label="Steps", value=4, precision=0)
                    video_cfg = gr.Number(label="Guidance", value=1.0)
                    video_megapixels = gr.Number(label="Megapixels", value=1.0)
                    video_seed = gr.Number(label="Seed", value=0, precision=0)
                    video_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    video_compile = gr.Checkbox(
                        label="torch.compile (faster after warmup; slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    video_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental — faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_frames_button = gr.Button("Edit all frames")
            extract_button.click(
                fn=_extract_video,
                inputs=[video_input, video_output, every_n, max_frames],
                outputs=[frame_state, frame_status],
            )
            edit_frames_button.click(
                fn=_edit_frames,
                inputs=[
                    frame_state,
                    video_prompt,
                    video_negative,
                    video_output,
                    video_steps,
                    video_cfg,
                    video_seed,
                    video_megapixels,
                    video_engine,
                    video_compile,
                    video_mrflow,
                ],
                outputs=frame_status,
            )

        with gr.Tab("Batch Folder"):
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(label="Input folder")
                    batch_output = gr.Textbox(label="Output folder", value=str(DEFAULT_OUTPUT_DIR))
                    batch_prompt = gr.Textbox(label="Prompt", lines=4)
                    batch_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    batch_steps = gr.Number(label="Steps", value=4, precision=0)
                    batch_cfg = gr.Number(label="Guidance", value=1.0)
                    batch_megapixels = gr.Number(label="Megapixels", value=1.0)
                    batch_seed = gr.Number(label="Seed", value=0, precision=0)
                    batch_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    batch_compile = gr.Checkbox(
                        label="torch.compile (faster after warmup; slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    batch_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental — faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    batch_button = gr.Button("Process folder")
                    batch_status = gr.Textbox(label="Status")
            batch_button.click(
                fn=_process_batch,
                inputs=[
                    batch_input,
                    batch_prompt,
                    batch_negative,
                    batch_output,
                    batch_steps,
                    batch_cfg,
                    batch_seed,
                    batch_megapixels,
                    batch_engine,
                    batch_compile,
                    batch_mrflow,
                ],
                outputs=batch_status,
            )

        with gr.Tab("Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Prompt", lines=4)
                    t2i_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    t2i_output = gr.Textbox(label="Output folder", value=str(DEFAULT_OUTPUT_DIR))
                    t2i_width = gr.Number(label="Width", value=1024, precision=0)
                    t2i_height = gr.Number(label="Height", value=1024, precision=0)
                    t2i_steps = gr.Number(label="Steps", value=4, precision=0)
                    t2i_cfg = gr.Number(label="Guidance", value=1.0)
                    t2i_seed = gr.Number(label="Seed", value=0, precision=0)
                    t2i_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    t2i_compile = gr.Checkbox(
                        label="torch.compile (faster after warmup; slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    t2i_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental — faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    t2i_button = gr.Button("Generate")
                    t2i_result = gr.Image(label="Result")
                    t2i_status = gr.Textbox(label="Status")
            t2i_button.click(
                fn=_generate_t2i,
                inputs=[
                    t2i_prompt,
                    t2i_negative,
                    t2i_output,
                    t2i_width,
                    t2i_height,
                    t2i_steps,
                    t2i_cfg,
                    t2i_seed,
                    t2i_engine,
                    t2i_compile,
                    t2i_mrflow,
                ],
                outputs=[t2i_result, t2i_status],
            )

    return demo


if __name__ == "__main__":
    launch_host = os.environ.get("COMFYUI_UI_HOST", "127.0.0.1")
    launch_port = int(os.environ.get("COMFYUI_UI_PORT", "7861"))
    build_app().launch(server_name=launch_host, server_port=launch_port, share=False)
