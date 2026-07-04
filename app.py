"""Simple Manga-to-Realistic Gradio app.

This entrypoint intentionally keeps startup small: no video tools, no audio
helpers, and no upscaler UI. It loads the diffusion pipeline only when an image
edit is requested.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from PIL import Image

import src.config  # noqa: F401 - configures cache paths and CUDA defaults
from src.config import BASE_DIR, DEFAULT_OPTIMIZATION_PROFILE
from src.core.image_gen import ImageGenerator
from src.core.pipeline_manager import PipelineManager
from src.runtime_policies import resolve_generation_guidance
from src.startup import ensure_cache_dirs
from src.utils.dimensions import calculate_dimensions_from_target_size


LOG_DIR = Path(BASE_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_CHOICES = [
    "Z-Image Turbo (Int8 - 8GB Safe)",
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
    "FLUX.2-klein-4B (Int8)",
]

DEFAULT_PROMPT = (
    "turn this manga or anime image into a realistic photo, natural skin texture, "
    "real-world lighting, detailed eyes, believable hair, preserve composition"
)
DEFAULT_NEGATIVE = (
    "anime, manga, drawing, illustration, cartoon, flat colors, distorted face, "
    "extra fingers, bad anatomy, blurry, low quality"
)

pipeline_manager = PipelineManager(BASE_DIR)
image_generator = ImageGenerator(pipeline_manager)


def _device_choices() -> list[str]:
    choices = []
    if torch.cuda.is_available():
        choices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        choices.append("mps")
    choices.append("cpu")
    return choices


def _fit_dimensions(image: Image.Image, long_edge: int) -> tuple[int, int]:
    width, height = image.size
    target_width, target_height = calculate_dimensions_from_target_size(
        width,
        height,
        int(long_edge),
    )
    target_width = max(16, (int(target_width) // 16) * 16)
    target_height = max(16, (int(target_height) // 16) * 16)
    return target_width, target_height


def manga_to_realistic(
    input_image: Optional[Image.Image],
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    long_edge: int,
    strength: float,
    steps: int,
    seed: int,
    device: str,
    use_builtin_realism_lora: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if input_image is None:
        return None, "Upload an image first."

    ensure_cache_dirs()
    input_image = input_image.convert("RGB")
    width, height = _fit_dimensions(input_image, long_edge)
    prompt = (prompt or DEFAULT_PROMPT).strip()
    negative_prompt = (negative_prompt or DEFAULT_NEGATIVE).strip()
    guidance = resolve_generation_guidance(model_choice, None)

    lora_file = None
    lora_strength = 0.7
    if use_builtin_realism_lora:
        progress(0.02, desc="Checking realism LoRA")
        lora_key = "zimage_realistic" if "Z-Image" in model_choice else "flux_anime2real"
        lora_file = pipeline_manager.ensure_builtin_lora_downloaded(lora_key, progress)
        if "FLUX" in model_choice:
            lora_strength = 1.0

    progress(0.05, desc="Loading image model")
    image, status, _ = image_generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=int(steps),
        seed=int(seed),
        guidance=guidance,
        device=device,
        model_choice=model_choice,
        input_images=[input_image],
        img2img_strength=float(strength),
        lora_file=lora_file,
        lora_strength=lora_strength,
        enable_preservation=False,
        enable_expression_transfer=False,
        enable_text_preservation=False,
        optimization_profile=DEFAULT_OPTIMIZATION_PROFILE,
        enable_optional_accelerators=False,
        progress_callback=progress,
    )
    return image, status


def unload_model() -> str:
    pipeline_manager.unload_pipeline()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded."


def build_app() -> gr.Blocks:
    devices = _device_choices()
    default_device = devices[0]
    with gr.Blocks(title="Manga to Realistic") as demo:
        gr.Markdown("# Manga to Realistic")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Manga image", type="pil", height=430)
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
                negative_prompt = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE, lines=3)
            with gr.Column(scale=1):
                output_image = gr.Image(label="Realistic output", type="pil", height=430)
                status = gr.Textbox(label="Status", lines=4)

        with gr.Row():
            model_choice = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
            device = gr.Dropdown(devices, value=default_device, label="Device")
            long_edge = gr.Slider(512, 1280, value=768, step=64, label="Long edge")

        with gr.Row():
            strength = gr.Slider(0.25, 0.95, value=0.72, step=0.01, label="Edit strength")
            steps = gr.Slider(1, 12, value=4, step=1, label="Steps")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            use_lora = gr.Checkbox(value=True, label="Realism LoRA")

        with gr.Row():
            run_btn = gr.Button("Generate", variant="primary")
            unload_btn = gr.Button("Unload model")

        run_btn.click(
            fn=manga_to_realistic,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                model_choice,
                long_edge,
                strength,
                steps,
                seed,
                device,
                use_lora,
            ],
            outputs=[output_image, status],
        )
        unload_btn.click(fn=unload_model, outputs=[status])

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("UFIG_GRADIO_PORT", "7860"))
    build_app().launch(
        server_name=os.environ.get("UFIG_SERVER_NAME", "127.0.0.1"),
        server_port=port,
        inbrowser=os.environ.get("UFIG_OPEN_BROWSER", "1") == "1",
    )
