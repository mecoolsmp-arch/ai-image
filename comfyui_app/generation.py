from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from pathlib import Path
from typing import Protocol

from comfyui_app.comfy_client import ComfyClient
from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT, get_hf_token
from comfyui_app.model_resolver import ModelResolverError, download_models, load_resolved_manifest, resolve_models
from comfyui_app.vram import detect_vram, select_tier
from comfyui_app.workflow_builder import build_edit_prompt, build_t2i_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    status: str
    prompt_id: str | None = None


class _ImageLike(Protocol):
    def save(self, fp: str | Path, format: str | None = None, **kwargs: object) -> object:
        ...


def _manifest_models(vram_gb: float, token: str | None, prefer_gguf: bool = False) -> dict[str, dict[str, object]]:
    manifest = load_resolved_manifest()
    if isinstance(manifest, dict):
        models = manifest.get("models")
        if isinstance(models, dict) and {"diffusion", "text_encoder", "vae"} <= set(models):
            cached = {
                "diffusion": dict(models["diffusion"]),
                "text_encoder": dict(models["text_encoder"]),
                "vae": dict(models["vae"]),
            }
            if not (prefer_gguf and not str(cached["diffusion"].get("local_filename", "")).lower().endswith(".gguf")):
                cached_paths = [Path(str(item.get("dest_dir", ""))) / str(item.get("local_filename", "")) for item in cached.values()]
                if all(path.exists() for path in cached_paths):
                    return cached
    resolved = resolve_models(vram_gb, token, prefer_gguf=prefer_gguf)
    return download_models(resolved, token)


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


def _resolved_filename_map(vram_gb: float, prefer_gguf: bool) -> dict[str, str]:
    resolved = _manifest_models(vram_gb, get_hf_token(), prefer_gguf=prefer_gguf)
    return {
        "diffusion": str(resolved["diffusion"]["local_filename"]),
        "text_encoder": str(resolved["text_encoder"]["local_filename"]),
        "vae": str(resolved["vae"]["local_filename"]),
    }


def _run_prompt(
    client: ComfyClient,
    prompt_dict: dict[str, object],
    output_dir: Path,
    output_name: str,
    timeout: float,
) -> GenerationResult:
    client.wait_until_up(timeout=timeout)
    prompt_id = client.queue_prompt(prompt_dict, client.client_id)
    client.wait_for_completion(prompt_id, client.client_id, timeout=timeout)
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
    client: ComfyClient | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    input_path = Path(input_image_path)
    target_dir = Path(output_dir)
    vram_gb, _, _ = detect_vram()
    tier = select_tier(vram_gb)
    filenames = _resolved_filename_map(vram_gb, prefer_gguf)
    uploaded_name = client.upload_image(input_path)
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
    )
    output_name = _output_name(input_path, "edit", seed)
    try:
        return _run_prompt(client, prompt_dict, target_dir, output_name, timeout)
    except Exception as exc:
        if not prefer_gguf and _retryable_oom(str(exc)):
            logger.warning("Switching to GGUF fallback after a memory error.")
            fallback = _resolved_filename_map(vram_gb, True)
            prompt_dict = build_edit_prompt(
                diffusion_model=fallback["diffusion"],
                text_encoder_model=fallback["text_encoder"],
                vae_model=fallback["vae"],
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
            )
            return _run_prompt(client, prompt_dict, target_dir, output_name, timeout)
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
    client: ComfyClient | None = None,
) -> GenerationResult:
    client = client or ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    target_dir = Path(output_dir)
    vram_gb, _, _ = detect_vram()
    tier = select_tier(vram_gb)
    filenames = _resolved_filename_map(vram_gb, prefer_gguf)
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
    )
    output_name = _output_name(None, "t2i", seed)
    try:
        return _run_prompt(client, prompt_dict, target_dir, output_name, timeout)
    except Exception as exc:
        if not prefer_gguf and _retryable_oom(str(exc)):
            logger.warning("Switching to GGUF fallback after a memory error.")
            fallback = _resolved_filename_map(vram_gb, True)
            prompt_dict = build_t2i_prompt(
                diffusion_model=fallback["diffusion"],
                text_encoder_model=fallback["text_encoder"],
                vae_model=fallback["vae"],
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
            )
            return _run_prompt(client, prompt_dict, target_dir, output_name, timeout)
        raise
