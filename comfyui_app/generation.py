from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from pathlib import Path
from typing import Callable, Protocol

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

from comfyui_app.comfy_client import ComfyClient
from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT, get_hf_token
from comfyui_app.model_resolver import (
    ModelResolverError,
    download_models,
    load_resolved_manifest,
    resolve_consistency_lora_models,
    resolve_depth_control_models,
    resolve_models,
)
from comfyui_app.vram import detect_vram, select_tier
from comfyui_app.workflow_builder import (
    DEFAULT_UPSCALE_MODEL,
    build_edit_prompt,
    build_esrgan_upscale_prompt,
    build_depth_refcontrol_edit_prompt,
    build_mrflow_edit_prompt,
    build_mrflow_t2i_prompt,
    build_rtx_upscale_prompt,
    build_t2i_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    status: str
    prompt_id: str | None = None
    preview_path: Path | None = None


class _ImageLike(Protocol):
    def save(self, fp: str | Path, format: str | None = None, **kwargs: object) -> object:
        ...


def _manifest_models(
    vram_gb: float,
    token: str | None,
    prefer_gguf: bool = False,
    engine: str = "int8",
) -> dict[str, dict[str, object]]:
    manifest = load_resolved_manifest()
    if isinstance(manifest, dict):
        if str(manifest.get("engine", "default")) != engine:
            manifest = None
    if isinstance(manifest, dict):
        models = manifest.get("models")
        if isinstance(models, dict) and {"diffusion", "text_encoder", "vae", "upscale"} <= set(models):
            cached = {
                "diffusion": dict(models["diffusion"]),
                "text_encoder": dict(models["text_encoder"]),
                "vae": dict(models["vae"]),
                "upscale": dict(models["upscale"]),
            }
            if not (prefer_gguf and not str(cached["diffusion"].get("local_filename", "")).lower().endswith(".gguf")):
                cached_paths = [Path(str(item.get("dest_dir", ""))) / str(item.get("local_filename", "")) for item in cached.values()]
                if all(path.exists() for path in cached_paths):
                    return cached
    resolved = resolve_models(vram_gb, token, prefer_gguf=prefer_gguf, engine=engine)
    return download_models(resolved, token, engine=engine)


def _output_name(input_image_path: Path | None, prefix: str, seed: int) -> str:
    stem = input_image_path.stem if input_image_path is not None else prefix
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or prefix
    return f"{safe_stem}_{prefix}_{seed}_{stamp}.png"


def _save_first_image(images: list[_ImageLike], output_dir: Path, output_name: str) -> Path:
    if not images:
        raise ModelResolverError("ComfyUI finished, but no image was returned.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    images[0].save(output_path)
    return output_path


def _retryable_oom(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "cuda out of memory" in lowered or "oom" in lowered


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    if Image is None:
        raise ModelResolverError("Pillow is required to read image dimensions for upscaling.")
    with Image.open(image_path) as image:
        width, height = image.size
    return width, height


def _depth_control_assets() -> tuple[str, str]:
    manifest = load_resolved_manifest()
    if not isinstance(manifest, dict):
        manifest = {}
    models = manifest.get("models")
    if not isinstance(models, dict):
        models = {}
    base = models.get("depth_control_base_int8")
    lora = models.get("depth_control_lora")
    if not isinstance(base, dict) or not isinstance(lora, dict):
        token = get_hf_token()
        if not token:
            raise ModelResolverError(
                "Pose/Shape lock (depth) is not installed. Re-run Install.bat with --with-depth-control."
            )
        resolved = resolve_depth_control_models(token)
        download_models(resolved, token, engine="int8")
        manifest = load_resolved_manifest()
        if not isinstance(manifest, dict):
            raise ModelResolverError(
                "Pose/Shape lock (depth) is not installed. Re-run Install.bat with --with-depth-control."
            )
        models = manifest.get("models")
        if not isinstance(models, dict):
            raise ModelResolverError(
                "Pose/Shape lock (depth) is not installed. Re-run Install.bat with --with-depth-control."
            )
        base = models.get("depth_control_base_int8")
        lora = models.get("depth_control_lora")
        if not isinstance(base, dict) or not isinstance(lora, dict):
            raise ModelResolverError(
                "Pose/Shape lock (depth) is not installed. Re-run Install.bat with --with-depth-control."
            )
    base_name = str(base.get("local_filename") or "")
    lora_name = str(lora.get("local_filename") or "")
    if not base_name or not lora_name:
        raise ModelResolverError(
            "Pose/Shape lock (depth) is not installed. Re-run Install.bat with --with-depth-control."
        )
    return base_name, lora_name


def _consistency_lora_asset() -> str:
    manifest = load_resolved_manifest()
    if not isinstance(manifest, dict):
        manifest = {}
    models = manifest.get("models")
    if not isinstance(models, dict):
        models = {}
    consistency = models.get("consistency_lora")
    if not isinstance(consistency, dict):
        token = get_hf_token()
        if not token:
            raise ModelResolverError(
                "Consistency LoRA is not installed. Re-run Install.bat with --with-consistency-lora."
            )
        resolved = resolve_consistency_lora_models(token)
        download_models(resolved, token, engine="int8")
        manifest = load_resolved_manifest()
        if not isinstance(manifest, dict):
            raise ModelResolverError(
                "Consistency LoRA is not installed. Re-run Install.bat with --with-consistency-lora."
            )
        models = manifest.get("models")
        if not isinstance(models, dict):
            raise ModelResolverError(
                "Consistency LoRA is not installed. Re-run Install.bat with --with-consistency-lora."
            )
        consistency = models.get("consistency_lora")
        if not isinstance(consistency, dict):
            raise ModelResolverError(
                "Consistency LoRA is not installed. Re-run Install.bat with --with-consistency-lora."
            )
    consistency_name = str(consistency.get("local_filename") or "")
    if not consistency_name:
        raise ModelResolverError(
            "Consistency LoRA is not installed. Re-run Install.bat with --with-consistency-lora."
        )
    return consistency_name


def _resolved_filename_map(vram_gb: float, prefer_gguf: bool, engine: str) -> dict[str, str]:
    resolved = _manifest_models(vram_gb, get_hf_token(), prefer_gguf=prefer_gguf, engine=engine)
    return {name: str(info["local_filename"]) for name, info in resolved.items()}


def _run_prompt(
    client: ComfyClient,
    prompt_dict: dict[str, object],
    output_dir: Path,
    output_name: str,
    timeout: float,
    preview_callback: Callable[[object], None] | None = None,
) -> GenerationResult:
    client.wait_until_up(timeout=timeout)
    prompt_id = client.queue_prompt(prompt_dict, client.client_id)
    client.wait_for_completion(prompt_id, client.client_id, timeout=timeout, preview_callback=preview_callback)
    images = client.get_images(prompt_id)
    image_path = _save_first_image(images, output_dir, output_name)
    return GenerationResult(image_path=image_path, status=f"Saved image to {image_path}.", prompt_id=prompt_id)


def run_edit(
    input_image_path: str | Path,
    prompt: str,
    negative: str,
    output_dir: str | Path,
    *,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = 0,
    megapixels: float = 1.0,
    batch_size: int = 1,
    decode_tile_size: int = 1024,
    use_tiled_decode: bool | None = None,
    timeout: float = 600.0,
    prefer_gguf: bool = False,
    engine: str = "int8",
    use_torch_compile: bool = False,
    use_consistency_lora: bool = False,
    consistency_lora_name: str | None = None,
    consistency_lora_strength: float = 1.0,
    mrflow: bool = False,
    mrflow_low_width: int = 512,
    mrflow_low_height: int = 512,
    mrflow_stage1_steps: int = 4,
    mrflow_refine_steps: int = 1,
    mrflow_refine_denoise: float = 0.25,
    mrflow_upscale_model_name: str = "RealESRGAN_x2plus.pth",
    client: ComfyClient | None = None,
    preview_callback: Callable[[object], None] | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    input_path = Path(input_image_path)
    target_dir = Path(output_dir)
    vram_gb, _, _ = detect_vram()
    tier = select_tier(vram_gb)
    filenames = _resolved_filename_map(vram_gb, prefer_gguf, engine)
    uploaded_name = client.upload_image(input_path)
    effective_consistency_lora = None
    if use_consistency_lora and not mrflow:
        effective_consistency_lora = consistency_lora_name or _consistency_lora_asset()
    if mrflow:
        prompt_dict = build_mrflow_edit_prompt(
            diffusion_model=filenames["diffusion"],
            text_encoder_model=filenames["text_encoder"],
            vae_model=filenames["vae"],
            upscale_model_name=filenames["upscale"],
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            stage1_steps=mrflow_stage1_steps,
            refine_steps=mrflow_refine_steps,
            refine_denoise=mrflow_refine_denoise,
            low_width=mrflow_low_width,
            low_height=mrflow_low_height,
            width=0,
            height=0,
            cfg=cfg,
            megapixels=megapixels,
            input_image_name=uploaded_name,
            batch_size=batch_size,
            use_tiled_decode=tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
            decode_tile_size=decode_tile_size,
            engine=engine,
            use_torch_compile=use_torch_compile,
        )
    else:
        prompt_dict = build_edit_prompt(
            diffusion_model=filenames["diffusion"],
            text_encoder_model=filenames["text_encoder"],
            vae_model=filenames["vae"],
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            megapixels=megapixels,
            input_image_name=uploaded_name,
            batch_size=batch_size,
            use_tiled_decode=tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
            decode_tile_size=decode_tile_size,
            engine=engine,
            use_torch_compile=use_torch_compile,
            consistency_lora_name=effective_consistency_lora,
            consistency_lora_strength=consistency_lora_strength,
        )
    output_name = _output_name(input_path, "edit", seed)
    try:
        return _run_prompt(client, prompt_dict, target_dir, output_name, timeout, preview_callback=preview_callback)
    except Exception as exc:
        if not prefer_gguf and _retryable_oom(str(exc)):
            logger.warning("Switching to GGUF fallback after a memory error.")
            fallback = _resolved_filename_map(vram_gb, True, engine)
            builder = build_mrflow_edit_prompt if mrflow else build_edit_prompt
            builder_kwargs: dict[str, object] = {
                "diffusion_model": fallback["diffusion"],
                "text_encoder_model": fallback["text_encoder"],
                "vae_model": fallback["vae"],
                "prompt": prompt,
                "negative": negative,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "megapixels": megapixels,
                "input_image_name": uploaded_name,
                "batch_size": batch_size,
                "use_tiled_decode": tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
                "decode_tile_size": decode_tile_size,
                "engine": engine,
                "use_torch_compile": use_torch_compile,
            }
            if not mrflow:
                builder_kwargs.update(
                    {
                        "consistency_lora_name": effective_consistency_lora,
                        "consistency_lora_strength": consistency_lora_strength,
                    }
                )
            if mrflow:
                builder_kwargs.update(
                    {
                        "upscale_model_name": fallback["upscale"],
                        "stage1_steps": mrflow_stage1_steps,
                        "refine_steps": mrflow_refine_steps,
                        "refine_denoise": mrflow_refine_denoise,
                        "low_width": mrflow_low_width,
                        "low_height": mrflow_low_height,
                        "width": 0,
                        "height": 0,
                    }
                )
            prompt_dict = builder(**builder_kwargs)
            return _run_prompt(client, prompt_dict, target_dir, output_name, timeout, preview_callback=preview_callback)
        raise


def run_t2i(
    prompt: str,
    negative: str,
    output_dir: str | Path,
    *,
    width: int,
    height: int,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = 0,
    batch_size: int = 1,
    decode_tile_size: int = 1024,
    use_tiled_decode: bool | None = None,
    timeout: float = 600.0,
    prefer_gguf: bool = False,
    use_teacache: bool = False,
    engine: str = "int8",
    use_torch_compile: bool = False,
    use_consistency_lora: bool = False,
    consistency_lora_name: str | None = None,
    consistency_lora_strength: float = 1.0,
    mrflow: bool = False,
    mrflow_low_width: int = 512,
    mrflow_low_height: int = 512,
    mrflow_stage1_steps: int = 4,
    mrflow_refine_steps: int = 1,
    mrflow_refine_denoise: float = 0.25,
    mrflow_upscale_model_name: str = "RealESRGAN_x2plus.pth",
    client: ComfyClient | None = None,
    preview_callback: Callable[[object], None] | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    target_dir = Path(output_dir)
    vram_gb, _, _ = detect_vram()
    tier = select_tier(vram_gb)
    filenames = _resolved_filename_map(vram_gb, prefer_gguf, engine)
    effective_consistency_lora = None
    if use_consistency_lora and not mrflow:
        effective_consistency_lora = consistency_lora_name or _consistency_lora_asset()
    if mrflow:
        prompt_dict = build_mrflow_t2i_prompt(
            diffusion_model=filenames["diffusion"],
            text_encoder_model=filenames["text_encoder"],
            vae_model=filenames["vae"],
            upscale_model_name=filenames["upscale"],
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            stage1_steps=mrflow_stage1_steps,
            refine_steps=mrflow_refine_steps,
            refine_denoise=mrflow_refine_denoise,
            low_width=mrflow_low_width,
            low_height=mrflow_low_height,
            width=width,
            height=height,
            cfg=cfg,
            batch_size=batch_size,
            use_tiled_decode=tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
            decode_tile_size=decode_tile_size,
            use_teacache=use_teacache,
            engine=engine,
            use_torch_compile=use_torch_compile,
        )
    else:
        prompt_dict = build_t2i_prompt(
            diffusion_model=filenames["diffusion"],
            text_encoder_model=filenames["text_encoder"],
            vae_model=filenames["vae"],
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            batch_size=batch_size,
            use_tiled_decode=tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
            decode_tile_size=decode_tile_size,
            engine=engine,
            use_torch_compile=use_torch_compile,
            consistency_lora_name=effective_consistency_lora,
            consistency_lora_strength=consistency_lora_strength,
        )
    output_name = _output_name(None, "t2i", seed)
    try:
        return _run_prompt(client, prompt_dict, target_dir, output_name, timeout, preview_callback=preview_callback)
    except Exception as exc:
        if not prefer_gguf and _retryable_oom(str(exc)):
            logger.warning("Switching to GGUF fallback after a memory error.")
            fallback = _resolved_filename_map(vram_gb, True, engine)
            builder = build_mrflow_t2i_prompt if mrflow else build_t2i_prompt
            builder_kwargs: dict[str, object] = {
                "diffusion_model": fallback["diffusion"],
                "text_encoder_model": fallback["text_encoder"],
                "vae_model": fallback["vae"],
                "prompt": prompt,
                "negative": negative,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "use_tiled_decode": tier.use_tiled_decode if use_tiled_decode is None else use_tiled_decode,
                "decode_tile_size": decode_tile_size,
                "use_teacache": use_teacache,
                "engine": engine,
                "use_torch_compile": use_torch_compile,
            }
            if not mrflow:
                builder_kwargs.update(
                    {
                        "consistency_lora_name": effective_consistency_lora,
                        "consistency_lora_strength": consistency_lora_strength,
                    }
                )
            if mrflow:
                builder_kwargs.update(
                    {
                        "upscale_model_name": fallback["upscale"],
                        "stage1_steps": mrflow_stage1_steps,
                        "refine_steps": mrflow_refine_steps,
                        "refine_denoise": mrflow_refine_denoise,
                        "low_width": mrflow_low_width,
                        "low_height": mrflow_low_height,
                    }
                )
            prompt_dict = builder(**builder_kwargs)
            return _run_prompt(client, prompt_dict, target_dir, output_name, timeout, preview_callback=preview_callback)
        raise


def run_upscale(
    input_image_path: str | Path,
    output_dir: str | Path,
    *,
    upscaler: str = "rtx",
    scale: float = 2.0,
    resize_type: str = "scale by multiplier",
    quality: str = "ULTRA",
    target_width: int | None = None,
    target_height: int | None = None,
    timeout: float = 600.0,
    client: ComfyClient | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    input_path = Path(input_image_path)
    target_dir = Path(output_dir)
    uploaded_name = client.upload_image(input_path)
    if resize_type == "scale by multiplier" and (target_width is None or target_height is None):
        source_width, source_height = _image_dimensions(input_path)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
    if upscaler == "rtx":
        if resize_type == "target dimensions":
            prompt_dict = build_rtx_upscale_prompt(
                image=uploaded_name,
                resize_type=resize_type,
                width=target_width,
                height=target_height,
                quality=quality,
            )
        else:
            prompt_dict = build_rtx_upscale_prompt(
                image=uploaded_name,
                resize_type=resize_type,
                scale=scale,
                quality=quality,
            )
    elif upscaler == "esrgan":
        prompt_dict = build_esrgan_upscale_prompt(
            image=uploaded_name,
            upscale_model_name=DEFAULT_UPSCALE_MODEL,
            resize_type="target dimensions" if target_width is not None and target_height is not None else resize_type,
            target_width=target_width,
            target_height=target_height,
        )
    else:
        raise ModelResolverError(f"Unknown upscaler: {upscaler}")

    output_name = _output_name(input_path, "upscale", 0)
    try:
        return _run_prompt(client, prompt_dict, target_dir, output_name, timeout)
    except Exception as exc:
        raise exc


def run_depth_edit(
    image_path: str | Path,
    reference_image_path: str | Path | None,
    prompt: str,
    negative: str,
    output_dir: str | Path,
    *,
    steps: int = 20,
    cfg: float = 5.0,
    seed: int = 0,
    lora_strength: float = 1.0,
    megapixels: float = 1.0,
    timeout: float = 600.0,
    use_teacache: bool = False,
    client: ComfyClient | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    depth_source_path = Path(image_path)
    reference_path = Path(reference_image_path) if reference_image_path is not None else depth_source_path
    target_dir = Path(output_dir)
    vram_gb, _, _ = detect_vram()
    filenames = _resolved_filename_map(vram_gb, False, "int8")
    depth_base_model, depth_lora_model = _depth_control_assets()
    depth_source_name = client.upload_image(depth_source_path)
    reference_name = client.upload_image(reference_path) if reference_path != depth_source_path else depth_source_name
    prompt_dict = build_depth_refcontrol_edit_prompt(
        diffusion_model=depth_base_model,
        text_encoder_model=filenames["text_encoder"],
        vae_model=filenames["vae"],
        lora_model_name=depth_lora_model,
        reference_image_name=reference_name,
        structure_image_name=depth_source_name,
        prompt=prompt,
        negative=negative,
        seed=seed,
        steps=steps,
        cfg=cfg,
        lora_strength=lora_strength,
        megapixels=megapixels,
        use_teacache=use_teacache,
    )
    output_name = _output_name(depth_source_path, "depth_edit", seed)
    client.wait_until_up(timeout=timeout)
    prompt_id = client.queue_prompt(prompt_dict, client.client_id)
    client.wait_for_completion(prompt_id, client.client_id, timeout=timeout)
    images = client.get_images(prompt_id)
    if len(images) < 2:
        raise ModelResolverError("ComfyUI finished, but the depth preview or final image was not returned.")
    target_dir.mkdir(parents=True, exist_ok=True)
    depth_preview_path = target_dir / output_name.replace(".png", "_depth.png")
    preview_image = images[0]
    final_image = images[-1]
    preview_image.save(depth_preview_path)
    final_path = target_dir / output_name
    final_image.save(final_path)
    return GenerationResult(
        image_path=final_path,
        status=f"Saved image to {final_path}.",
        prompt_id=prompt_id,
        preview_path=depth_preview_path,
    )
