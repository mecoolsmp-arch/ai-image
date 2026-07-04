from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import gradio as gr
except Exception:  # pragma: no cover - optional dependency
    gr = None  # type: ignore[assignment]

from comfyui_app.batch import CANCEL_EVENT, clear_cancel, iter_process_folder, process_folder, request_cancel
from comfyui_app.comfy_client import ComfyClient
from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT, DEFAULT_OUTPUT_DIR
from comfyui_app.generation import GenerationResult, run_depth_edit, run_edit, run_t2i, run_upscale
from comfyui_app.model_resolver import ModelResolverError, load_resolved_manifest
from comfyui_app.ui_utils import pick_directory
from comfyui_app.vram import detect_vram, select_tier
from comfyui_app.video_frames import extract_frames, frames_to_video, probe_video_metadata

logger = logging.getLogger(__name__)

ENGINE_CHOICES = [
    ("fp8 (default)", "default"),
    ("Nunchaku INT4 (experimental - faster, needs extra install)", "nunchaku_int4"),
]
UPSCALER_CHOICES = [
    ("NVIDIA RTX VSR (experimental)", "rtx"),
    ("Real-ESRGAN x2+", "esrgan"),
]
QUALITY_CHOICES = [("LOW", "LOW"), ("MEDIUM", "MEDIUM"), ("HIGH", "HIGH"), ("ULTRA", "ULTRA")]
VIDEO_EXTS = [".mp4", ".mov"]


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, ModelResolverError):
        return exc.message
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _component_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "filepath", "tempfile"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return Path(item)
    return Path(str(value))


def _optional_component_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    try:
        return _component_path(value)
    except Exception:
        return None


def _status_markdown() -> str:
    manifest = load_resolved_manifest()
    if not isinstance(manifest, dict):
        return "Ready. No resolved model manifest found yet."
    models = manifest.get("models")
    engine = manifest.get("engine", "default")
    if not isinstance(models, dict):
        return "Ready. The resolved model manifest is present, but it is incomplete."
    lines = [f"Ready. Current manifest engine: `{engine}`."]
    for key in ("diffusion", "text_encoder", "vae", "upscale"):
        entry = models.get(key)
        if isinstance(entry, dict):
            filename = entry.get("local_filename", "")
            lines.append(f"- {key}: `{filename}`")
    return "\n".join(lines)


def refresh_status() -> str:
    try:
        vram_gb, device_name, cuda_available = detect_vram()
        tier = select_tier(vram_gb)
        device_line = f"Detected {device_name} ({vram_gb:.1f} GB VRAM)" if cuda_available else "No CUDA GPU detected"
        return f"**{device_line}**\n\nUsing the **{tier.label}** setup.\n\n{_status_markdown()}"
    except Exception as exc:
        return _friendly_error(exc)


def _stop_current_job() -> str:
    request_cancel()
    try:
        ComfyClient(COMFYUI_HOST, COMFYUI_PORT).interrupt()
    except Exception:
        logger.debug("Interrupt request failed", exc_info=True)
    return "Stop requested."


def _edit_handler(
    input_image: object,
    reference_image: object,
    prompt: str,
    negative: str,
    output_dir: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
    depth_lock: bool,
) -> tuple[str | None, str]:
    try:
        input_path = _component_path(input_image)
        reference_path = _optional_component_path(reference_image)
        if depth_lock:
            result = run_depth_edit(
                input_path,
                reference_path,
                prompt,
                negative,
                output_dir,
                seed=int(seed),
            )
        else:
            result = run_edit(
                input_path,
                prompt,
                negative,
                output_dir,
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


def _t2i_handler(
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


def _upscale_handler(
    input_image: object,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
) -> tuple[str | None, str]:
    try:
        result = run_upscale(
            _component_path(input_image),
            output_dir,
            upscaler=upscaler,
            scale=float(scale),
            quality=quality,
        )
        return str(result.image_path), result.status
    except Exception as exc:
        return None, _friendly_error(exc)


def _yield_folder_updates(summary_iter):
    last_state: tuple[object, ...] | None = None
    for summary in summary_iter:
        results = tuple(summary.get("results", []))
        failures = tuple(summary.get("failures", []))
        state = (summary.get("count"), results, failures, summary.get("message"), summary.get("output_dir"))
        if state == last_state:
            continue
        last_state = state
        yield summary.get("message", ""), list(results)


def _batch_folder_stream(
    input_dir: str,
    output_dir: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
):
    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        return run_edit(
            image_path,
            prompt_text,
            negative_text,
            run_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )

    return _yield_folder_updates(iter_process_folder(input_dir, output_dir, prompt, negative, gen_fn))


def _upscale_folder_stream(
    input_dir: str,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
):
    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        return run_upscale(image_path, run_dir, upscaler=upscaler, scale=float(scale), quality=quality)

    return _yield_folder_updates(iter_process_folder(input_dir, output_dir, "", "", gen_fn))


def _extract_frames_handler(video_file: object, output_dir: str, every_n: int, max_frames: int) -> tuple[str, str]:
    try:
        video_path = _component_path(video_file)
        run_dir = Path(output_dir) / f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved = extract_frames(video_path, run_dir, every_n=int(every_n), max_frames=int(max_frames) or None)
        return str(run_dir), f"Extracted {len(saved)} frames to {run_dir}."
    except Exception as exc:
        return output_dir, _friendly_error(exc)


def _edit_frames_handler(
    input_dir: str,
    output_dir: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> tuple[str]:
    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        return run_edit(
            image_path,
            prompt_text,
            negative_text,
            run_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
        )

    try:
        summary = process_folder(input_dir, output_dir, prompt, negative, gen_fn)
        return (summary["message"],)
    except Exception as exc:
        return (_friendly_error(exc),)


def _validate_video_file(video_file: object) -> Path:
    video_path = _component_path(video_file)
    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ModelResolverError("Video Upscale only accepts .mp4 and .mov input files.")
    return video_path


def _video_upscale_handler(
    video_file: object,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
) -> tuple[str | None, str]:
    try:
        clear_cancel()
        source_path = _validate_video_file(video_file)
        metadata = probe_video_metadata(source_path)
        fps = float(metadata.get("fps") or 0.0) or 24.0
        source_width = int(metadata.get("width") or 0)
        source_height = int(metadata.get("height") or 0)
        has_audio = bool(metadata.get("has_audio"))
        target_width = max(1, int(round(source_width * float(scale)))) if source_width > 0 else None
        target_height = max(1, int(round(source_height * float(scale)))) if source_height > 0 else None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{source_path.stem}_upscaled.mp4"
        with TemporaryDirectory(dir=str(output_root)) as work_dir:
            work_root = Path(work_dir)
            frames_dir = work_root / "frames"
            upscaled_dir = work_root / "upscaled"
            extracted_frames = extract_frames(source_path, frames_dir)
            if CANCEL_EVENT.is_set():
                return None, "Cancelled."
            upscaled_dir.mkdir(parents=True, exist_ok=True)
            for index, frame_path in enumerate(extracted_frames):
                if CANCEL_EVENT.is_set():
                    return None, f"Cancelled after {index} frames."
                result = run_upscale(
                    frame_path,
                    upscaled_dir,
                    upscaler=upscaler,
                    scale=float(scale),
                    quality=quality,
                    target_width=target_width,
                    target_height=target_height,
                )
                destination = upscaled_dir / f"frame_{index:06d}.png"
                shutil.copy2(result.image_path, destination)
            if CANCEL_EVENT.is_set():
                return None, "Cancelled."
            final_path = frames_to_video(upscaled_dir, output_path, fps, audio_source=source_path if has_audio else None)
            return str(final_path), f"Saved video to {final_path}."
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

        def directory_field(label: str, value: str) -> tuple[object, object]:
            with gr.Row():
                textbox = gr.Textbox(label=label, value=value, scale=6)
                browse_button = gr.Button("Browse...", scale=1)
            browse_button.click(fn=pick_directory, inputs=[textbox], outputs=[textbox])
            return textbox, browse_button

        with gr.Tab("Image Edit"):
            with gr.Row():
                with gr.Column():
                    edit_image = gr.Image(label="Depth source image", type="filepath")
                    edit_reference = gr.Image(label="Reference / identity image (optional)", type="filepath")
                    edit_prompt = gr.Textbox(label="Prompt", lines=4)
                    edit_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    edit_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                    edit_steps = gr.Number(label="Steps", value=4, precision=0)
                    edit_cfg = gr.Number(label="Guidance", value=1.0)
                    edit_megapixels = gr.Number(label="Megapixels", value=1.0)
                    edit_seed = gr.Number(label="Seed", value=0, precision=0)
                    edit_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    edit_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    edit_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_depth_lock = gr.Checkbox(
                        label="Pose/Shape lock (depth reference) — experimental, slower (~20 steps, base model)",
                        value=False,
                    )
                    edit_button = gr.Button("Generate")
                    edit_stop = gr.Button("Stop")
                    edit_result = gr.Image(label="Result")
                    edit_status = gr.Textbox(label="Status")
            edit_run = edit_button.click(
                fn=_edit_handler,
                inputs=[edit_image, edit_reference, edit_prompt, edit_negative, edit_output, edit_steps, edit_cfg, edit_megapixels, edit_seed, edit_engine, edit_compile, edit_mrflow, edit_depth_lock],
                outputs=[edit_result, edit_status],
            )
            edit_stop.click(fn=_stop_current_job, outputs=edit_status, cancels=[edit_run])

        with gr.Tab("Video to Frames"):
            frame_state = gr.State("")
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video")
                    every_n = gr.Number(label="Every Nth frame", value=1, precision=0)
                    max_frames = gr.Number(label="Max frames", value=0, precision=0)
                    video_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                    extract_button = gr.Button("Extract frames")
                    frame_status = gr.Textbox(label="Status")
                with gr.Column():
                    frame_dir = gr.Textbox(label="Extracted frames folder", value=str(DEFAULT_OUTPUT_DIR))
                    edit_prompt = gr.Textbox(label="Prompt", lines=4)
                    edit_negative = gr.Textbox(label="Negative prompt", lines=3)
                    edit_steps = gr.Number(label="Steps", value=4, precision=0)
                    edit_cfg = gr.Number(label="Guidance", value=1.0)
                    edit_megapixels = gr.Number(label="Megapixels", value=1.0)
                    edit_seed = gr.Number(label="Seed", value=0, precision=0)
                    edit_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    edit_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    edit_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_frames_button = gr.Button("Edit all frames")
                    edit_frames_stop = gr.Button("Stop")
            extract_button.click(
                fn=_extract_frames_handler,
                inputs=[video_input, video_output, every_n, max_frames],
                outputs=[frame_dir, frame_status],
            )
            edit_frames_evt = edit_frames_button.click(
                fn=_edit_frames_handler,
                inputs=[frame_dir, video_output, edit_prompt, edit_negative, edit_steps, edit_cfg, edit_megapixels, edit_seed, edit_engine, edit_compile, edit_mrflow],
                outputs=frame_status,
            )
            edit_frames_stop.click(fn=_stop_current_job, outputs=frame_status, cancels=[edit_frames_evt])

        with gr.Tab("Batch Folder"):
            with gr.Row():
                with gr.Column():
                    batch_input, _ = directory_field("Input folder", str(DEFAULT_OUTPUT_DIR))
                    batch_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                    batch_prompt = gr.Textbox(label="Prompt", lines=4)
                    batch_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    batch_steps = gr.Number(label="Steps", value=4, precision=0)
                    batch_cfg = gr.Number(label="Guidance", value=1.0)
                    batch_megapixels = gr.Number(label="Megapixels", value=1.0)
                    batch_seed = gr.Number(label="Seed", value=0, precision=0)
                    batch_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    batch_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    batch_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    batch_button = gr.Button("Process folder")
                    batch_stop = gr.Button("Stop")
                    batch_status = gr.Textbox(label="Status")
                    batch_gallery = gr.Gallery(label="Results", columns=3, height=240)
            batch_evt = batch_button.click(
                fn=_batch_folder_stream,
                inputs=[batch_input, batch_output, batch_prompt, batch_negative, batch_steps, batch_cfg, batch_megapixels, batch_seed, batch_engine, batch_compile, batch_mrflow],
                outputs=[batch_status, batch_gallery],
            )
            batch_stop.click(fn=_stop_current_job, outputs=batch_status, cancels=[batch_evt])

        with gr.Tab("Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Prompt", lines=4)
                    t2i_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    t2i_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                    t2i_width = gr.Number(label="Width", value=1024, precision=0)
                    t2i_height = gr.Number(label="Height", value=1024, precision=0)
                    t2i_steps = gr.Number(label="Steps", value=4, precision=0)
                    t2i_cfg = gr.Number(label="Guidance", value=1.0)
                    t2i_seed = gr.Number(label="Seed", value=0, precision=0)
                    t2i_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value="default")
                    t2i_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    t2i_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    t2i_button = gr.Button("Generate")
                    t2i_stop = gr.Button("Stop")
                    t2i_result = gr.Image(label="Result")
                    t2i_status = gr.Textbox(label="Status")
            t2i_evt = t2i_button.click(
                fn=_t2i_handler,
                inputs=[t2i_prompt, t2i_negative, t2i_output, t2i_width, t2i_height, t2i_steps, t2i_cfg, t2i_seed, t2i_engine, t2i_compile, t2i_mrflow],
                outputs=[t2i_result, t2i_status],
            )
            t2i_stop.click(fn=_stop_current_job, outputs=t2i_status, cancels=[t2i_evt])

        with gr.Tab("Upscale"):
            with gr.Tabs():
                with gr.Tab("Single image"):
                    with gr.Row():
                        with gr.Column():
                            upscale_image = gr.Image(label="Image", type="filepath")
                            upscale_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                        with gr.Column():
                            upscale_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                            upscale_scale = gr.Number(label="Scale", value=2.0, precision=2)
                            upscale_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                            upscale_button = gr.Button("Generate")
                            upscale_stop = gr.Button("Stop")
                            upscale_result = gr.Image(label="Result")
                            upscale_status = gr.Textbox(label="Status")
                    upscale_evt = upscale_button.click(
                        fn=_upscale_handler,
                        inputs=[upscale_image, upscale_output, upscale_upscaler, upscale_scale, upscale_quality],
                        outputs=[upscale_result, upscale_status],
                    )
                    upscale_stop.click(fn=_stop_current_job, outputs=upscale_status, cancels=[upscale_evt])

                with gr.Tab("Folder"):
                    with gr.Row():
                        with gr.Column():
                            upscale_input_folder, _ = directory_field("Input folder", str(DEFAULT_OUTPUT_DIR))
                            upscale_folder_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                        with gr.Column():
                            upscale_folder_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                            upscale_folder_scale = gr.Number(label="Scale", value=2.0, precision=2)
                            upscale_folder_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                            upscale_folder_button = gr.Button("Process folder")
                            upscale_folder_stop = gr.Button("Stop")
                            upscale_folder_status = gr.Textbox(label="Status")
                            upscale_folder_gallery = gr.Gallery(label="Results", columns=3, height=240)
                    upscale_folder_evt = upscale_folder_button.click(
                        fn=_upscale_folder_stream,
                        inputs=[upscale_input_folder, upscale_folder_output, upscale_folder_upscaler, upscale_folder_scale, upscale_folder_quality],
                        outputs=[upscale_folder_status, upscale_folder_gallery],
                    )
                    upscale_folder_stop.click(fn=_stop_current_job, outputs=upscale_folder_status, cancels=[upscale_folder_evt])

        with gr.Tab("Video Upscale"):
            with gr.Row():
                with gr.Column():
                    video_upscale_input = gr.File(label="Video file", file_count="single", file_types=VIDEO_EXTS, type="filepath")
                    video_upscale_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR))
                    video_upscale_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                    video_upscale_scale = gr.Number(label="Scale", value=2.0, precision=2)
                    video_upscale_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                    video_upscale_button = gr.Button("Upscale")
                    video_upscale_stop = gr.Button("Stop")
                    video_upscale_status = gr.Textbox(label="Status")
                with gr.Column():
                    video_upscale_result = gr.Video(label="Output video")
            video_upscale_evt = video_upscale_button.click(
                fn=_video_upscale_handler,
                inputs=[video_upscale_input, video_upscale_output, video_upscale_upscaler, video_upscale_scale, video_upscale_quality],
                outputs=[video_upscale_result, video_upscale_status],
            )
            video_upscale_stop.click(fn=_stop_current_job, outputs=video_upscale_status, cancels=[video_upscale_evt])

        demo.queue()
    return demo


if __name__ == "__main__":
    launch_host = os.environ.get("COMFYUI_UI_HOST", "127.0.0.1")
    launch_port = int(os.environ.get("COMFYUI_UI_PORT", "7861"))
    build_app().launch(server_name=launch_host, server_port=launch_port, share=False)
