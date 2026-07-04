from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from comfyui_app.config import WORKFLOWS_DIR


TORCH_COMPILE_NODE_CLASS = "TorchCompileModel"
# Verified from nunchaku-ai/ComfyUI-nunchaku nodes/models/flux.py:
# class_type / mapping key is "NunchakuFluxDiTLoader".
NUNCHAKU_DIT_LOADER_CLASS = "NunchakuFluxDiTLoader"
BASE_FLUX2_KLEIN_FP8_MODEL = "flux-2-klein-base-4b-fp8.safetensors"
DEPTH_REFCONTROL_LORA_MODEL = "flux2_klein_4b_refcontrol_depth.safetensors"
DEPTH_ANYTHING_V2_PREPROCESSOR_CLASS = "DepthAnythingV2Preprocessor"
LORA_LOADER_MODEL_ONLY_CLASS = "LoraLoaderModelOnly"
REFERENCE_LATENT_CLASS = "ReferenceLatent"
REFCONTROL_TRIGGER = "refcontrol"
UPSCALE_MODEL_LOADER_CLASS = "UpscaleModelLoader"
IMAGE_UPSCALE_WITH_MODEL_CLASS = "ImageUpscaleWithModel"
SPLIT_SIGMAS_DENOISE_CLASS = "SplitSigmasDenoise"
RTX_VIDEO_SUPER_RESOLUTION_CLASS = "RTXVideoSuperResolution"
DEFAULT_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _link(node_id: str, output_index: int = 0) -> list[Any]:
    return [node_id, output_index]


def _loader_node(diffusion_model: str) -> dict[str, Any]:
    if diffusion_model.lower().endswith(".gguf"):
        return _node("UnetLoaderGGUF", unet_name=diffusion_model)
    return _node("UNETLoader", unet_name=diffusion_model, weight_dtype="default")


def _diffusion_loader_node(diffusion_model: str, engine: str) -> dict[str, Any]:
    if engine == "nunchaku_int4":
        return _node(
            NUNCHAKU_DIT_LOADER_CLASS,
            model_path=diffusion_model,
            cache_threshold=0.12,
            attention="nunchaku-fp16",
            cpu_offload="auto",
            device_id=0,
            data_type="float16",
        )
    return _loader_node(diffusion_model)


def _decode_node(use_tiled_decode: bool, decode_tile_size: int) -> tuple[str, dict[str, Any]]:
    if use_tiled_decode:
        return "18", _node(
            "VAEDecodeTiled",
            samples=_link("17"),
            vae=_link("3"),
            tile_size=decode_tile_size,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
        )
    return "18", _node("VAEDecode", samples=_link("17"), vae=_link("3"))


def _apply_torch_compile(nodes: dict[str, Any]) -> None:
    compile_id = str(max(int(node_id) for node_id in nodes.keys() if node_id.isdigit()) + 1)
    nodes[compile_id] = _node(TORCH_COMPILE_NODE_CLASS, model=_link("1"), backend="inductor")
    for node in nodes.values():
        if isinstance(node, dict) and node.get("class_type") == "CFGGuider":
            node["inputs"]["model"] = _link(compile_id)


def build_edit_prompt(
    *,
    diffusion_model: str,
    text_encoder_model: str,
    vae_model: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    megapixels: float,
    input_image_name: str,
    batch_size: int,
    use_tiled_decode: bool,
    decode_tile_size: int,
    engine: str = "default",
    use_torch_compile: bool = False,
) -> dict[str, Any]:
    model_link = _link("1")
    nodes: dict[str, Any] = {
        "1": _diffusion_loader_node(diffusion_model, engine),
        "2": _node("CLIPLoader", clip_name=text_encoder_model, type="flux2", device="default"),
        "3": _node("VAELoader", vae_name=vae_model),
        "4": _node("CLIPTextEncode", clip=_link("2"), text=prompt),
        "5": _node("CLIPTextEncode", clip=_link("2"), text=negative),
        "6": _node("LoadImage", image=input_image_name),
        "7": _node(
            "ImageScaleToTotalPixels",
            image=_link("6"),
            upscale_method="nearest-exact",
            megapixels=megapixels,
            resolution_steps=1,
        ),
        "8": _node("GetImageSize", image=_link("7")),
        "9": _node("VAEEncode", pixels=_link("7"), vae=_link("3")),
        "10": _node("ReferenceLatent", conditioning=_link("4"), latent=_link("9")),
        "11": _node("ReferenceLatent", conditioning=_link("5"), latent=_link("9")),
        "12": _node("EmptyFlux2LatentImage", width=_link("8", 0), height=_link("8", 1), batch_size=batch_size),
        "13": _node("Flux2Scheduler", steps=steps, width=_link("8", 0), height=_link("8", 1)),
        "14": _node("KSamplerSelect", sampler_name="euler"),
        "15": _node("RandomNoise", noise_seed=seed),
        "16": _node("CFGGuider", model=model_link, positive=_link("10"), negative=_link("11"), cfg=cfg),
        "17": _node(
            "SamplerCustomAdvanced",
            noise=_link("15"),
            guider=_link("16"),
            sampler=_link("14"),
            sigmas=_link("13"),
            latent_image=_link("12"),
        ),
    }
    decode_id, decode_node = _decode_node(use_tiled_decode, decode_tile_size)
    nodes[decode_id] = decode_node
    nodes["19"] = _node("SaveImage", images=_link(decode_id), filename_prefix="Flux2-Klein")
    if use_torch_compile:
        _apply_torch_compile(nodes)
    return nodes


def build_t2i_prompt(
    *,
    diffusion_model: str,
    text_encoder_model: str,
    vae_model: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    batch_size: int,
    use_tiled_decode: bool,
    decode_tile_size: int,
    engine: str = "default",
    use_torch_compile: bool = False,
) -> dict[str, Any]:
    model_link = _link("1")
    nodes: dict[str, Any] = {
        "1": _diffusion_loader_node(diffusion_model, engine),
        "2": _node("CLIPLoader", clip_name=text_encoder_model, type="flux2", device="default"),
        "3": _node("VAELoader", vae_name=vae_model),
        "4": _node("CLIPTextEncode", clip=_link("2"), text=prompt),
        "5": _node("CLIPTextEncode", clip=_link("2"), text=negative),
        "6": _node("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch_size),
        "7": _node("Flux2Scheduler", steps=steps, width=width, height=height),
        "8": _node("KSamplerSelect", sampler_name="euler"),
        "9": _node("RandomNoise", noise_seed=seed),
        "10": _node("CFGGuider", model=model_link, positive=_link("4"), negative=_link("5"), cfg=cfg),
        "11": _node(
            "SamplerCustomAdvanced",
            noise=_link("9"),
            guider=_link("10"),
            sampler=_link("8"),
            sigmas=_link("7"),
            latent_image=_link("6"),
        ),
    }
    decode_id, decode_node = _decode_node(use_tiled_decode, decode_tile_size)
    nodes[decode_id] = decode_node
    nodes["12"] = _node("SaveImage", images=_link(decode_id), filename_prefix="Flux2-Klein")
    if use_torch_compile:
        _apply_torch_compile(nodes)
    return nodes


def build_mrflow_t2i_prompt(
    *,
    diffusion_model: str,
    text_encoder_model: str,
    vae_model: str,
    upscale_model_name: str,
    prompt: str,
    negative: str,
    seed: int,
    stage1_steps: int,
    refine_steps: int,
    refine_denoise: float,
    low_width: int,
    low_height: int,
    width: int,
    height: int,
    cfg: float,
    batch_size: int,
    use_tiled_decode: bool,
    decode_tile_size: int,
    steps: int | None = None,
    engine: str = "default",
    use_torch_compile: bool = False,
) -> dict[str, Any]:
    model_link = _link("1")
    target_megapixels = (width * height) / 1_048_576
    nodes: dict[str, Any] = {
        "1": _diffusion_loader_node(diffusion_model, engine),
        "2": _node("CLIPLoader", clip_name=text_encoder_model, type="flux2", device="default"),
        "3": _node("VAELoader", vae_name=vae_model),
        "4": _node("CLIPTextEncode", clip=_link("2"), text=prompt),
        "5": _node("CLIPTextEncode", clip=_link("2"), text=negative),
        "6": _node("EmptyFlux2LatentImage", width=low_width, height=low_height, batch_size=batch_size),
        "7": _node("Flux2Scheduler", steps=stage1_steps, width=low_width, height=low_height),
        "8": _node("KSamplerSelect", sampler_name="euler"),
        "9": _node("RandomNoise", noise_seed=seed),
        "10": _node("CFGGuider", model=model_link, positive=_link("4"), negative=_link("5"), cfg=cfg),
        "11": _node(
            "SamplerCustomAdvanced",
            noise=_link("9"),
            guider=_link("10"),
            sampler=_link("8"),
            sigmas=_link("7"),
            latent_image=_link("6"),
        ),
        "12": _node("VAEDecode", samples=_link("11"), vae=_link("3")),
        "13": _node(UPSCALE_MODEL_LOADER_CLASS, model_name=upscale_model_name),
        "14": _node(IMAGE_UPSCALE_WITH_MODEL_CLASS, upscale_model=_link("13"), image=_link("12")),
        "15": _node(
            "ImageScaleToTotalPixels",
            image=_link("14"),
            upscale_method="lanczos",
            megapixels=target_megapixels,
            resolution_steps=1,
        ),
        "16": _node("GetImageSize", image=_link("15")),
        "17": _node("VAEEncode", pixels=_link("15"), vae=_link("3")),
        "18": _node("Flux2Scheduler", steps=stage1_steps + refine_steps, width=_link("16", 0), height=_link("16", 1)),
        "19": _node(SPLIT_SIGMAS_DENOISE_CLASS, sigmas=_link("18"), denoise=refine_denoise),
        "20": _node("KSamplerSelect", sampler_name="euler"),
        "21": _node("RandomNoise", noise_seed=seed + 1),
        "22": _node("CFGGuider", model=model_link, positive=_link("4"), negative=_link("5"), cfg=cfg),
        "23": _node(
            "SamplerCustomAdvanced",
            noise=_link("21"),
            guider=_link("22"),
            sampler=_link("20"),
            sigmas=_link("19", 1),
            latent_image=_link("17"),
        ),
        "24": _node(
            "VAEDecodeTiled" if use_tiled_decode else "VAEDecode",
            samples=_link("23"),
            vae=_link("3"),
            **(
                {
                    "tile_size": decode_tile_size,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                }
                if use_tiled_decode
                else {}
            ),
        ),
        "25": _node("SaveImage", images=_link("24"), filename_prefix="Flux2-Klein-MrFlow"),
    }
    if use_torch_compile:
        _apply_torch_compile(nodes)
    return nodes


def build_mrflow_edit_prompt(
    *,
    diffusion_model: str,
    text_encoder_model: str,
    vae_model: str,
    upscale_model_name: str,
    prompt: str,
    negative: str,
    seed: int,
    stage1_steps: int,
    refine_steps: int,
    refine_denoise: float,
    low_width: int,
    low_height: int,
    width: int,
    height: int,
    cfg: float,
    megapixels: float,
    input_image_name: str,
    batch_size: int,
    use_tiled_decode: bool,
    decode_tile_size: int,
    steps: int | None = None,
    engine: str = "default",
    use_torch_compile: bool = False,
) -> dict[str, Any]:
    model_link = _link("1")
    low_megapixels = (low_width * low_height) / 1_048_576
    target_megapixels = (width * height) / 1_048_576 if width > 0 and height > 0 else megapixels
    nodes: dict[str, Any] = {
        "1": _diffusion_loader_node(diffusion_model, engine),
        "2": _node("CLIPLoader", clip_name=text_encoder_model, type="flux2", device="default"),
        "3": _node("VAELoader", vae_name=vae_model),
        "4": _node("CLIPTextEncode", clip=_link("2"), text=prompt),
        "5": _node("CLIPTextEncode", clip=_link("2"), text=negative),
        "6": _node("LoadImage", image=input_image_name),
        "7": _node(
            "ImageScaleToTotalPixels",
            image=_link("6"),
            upscale_method="nearest-exact",
            megapixels=low_megapixels,
            resolution_steps=1,
        ),
        "8": _node("GetImageSize", image=_link("7")),
        "9": _node("VAEEncode", pixels=_link("7"), vae=_link("3")),
        "10": _node("ReferenceLatent", conditioning=_link("4"), latent=_link("9")),
        "11": _node("ReferenceLatent", conditioning=_link("5"), latent=_link("9")),
        "12": _node("EmptyFlux2LatentImage", width=_link("8", 0), height=_link("8", 1), batch_size=batch_size),
        "13": _node("Flux2Scheduler", steps=stage1_steps, width=_link("8", 0), height=_link("8", 1)),
        "14": _node("KSamplerSelect", sampler_name="euler"),
        "15": _node("RandomNoise", noise_seed=seed),
        "16": _node("CFGGuider", model=model_link, positive=_link("10"), negative=_link("11"), cfg=cfg),
        "17": _node(
            "SamplerCustomAdvanced",
            noise=_link("15"),
            guider=_link("16"),
            sampler=_link("14"),
            sigmas=_link("13"),
            latent_image=_link("12"),
        ),
        "18": _node("VAEDecode", samples=_link("17"), vae=_link("3")),
        "19": _node(UPSCALE_MODEL_LOADER_CLASS, model_name=upscale_model_name),
        "20": _node(IMAGE_UPSCALE_WITH_MODEL_CLASS, upscale_model=_link("19"), image=_link("18")),
        "21": _node(
            "ImageScaleToTotalPixels",
            image=_link("20"),
            upscale_method="lanczos",
            megapixels=target_megapixels,
            resolution_steps=1,
        ),
        "22": _node("GetImageSize", image=_link("21")),
        "23": _node("VAEEncode", pixels=_link("21"), vae=_link("3")),
        "24": _node("Flux2Scheduler", steps=stage1_steps + refine_steps, width=_link("22", 0), height=_link("22", 1)),
        "25": _node(SPLIT_SIGMAS_DENOISE_CLASS, sigmas=_link("24"), denoise=refine_denoise),
        "26": _node("KSamplerSelect", sampler_name="euler"),
        "27": _node("RandomNoise", noise_seed=seed + 1),
        "28": _node("CFGGuider", model=model_link, positive=_link("4"), negative=_link("5"), cfg=cfg),
        "29": _node(
            "SamplerCustomAdvanced",
            noise=_link("27"),
            guider=_link("28"),
            sampler=_link("26"),
            sigmas=_link("25", 1),
            latent_image=_link("23"),
        ),
    }
    nodes["30"] = _node(
        "VAEDecodeTiled" if use_tiled_decode else "VAEDecode",
        samples=_link("29"),
        vae=_link("3"),
        **(
            {
                "tile_size": decode_tile_size,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            }
            if use_tiled_decode
            else {}
        ),
    )
    nodes["31"] = _node("SaveImage", images=_link("30"), filename_prefix="Flux2-Klein-MrFlow")
    if use_torch_compile:
        _apply_torch_compile(nodes)
    return nodes


def _ensure_refcontrol_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if REFCONTROL_TRIGGER in lowered:
        return prompt
    return f"{REFCONTROL_TRIGGER}, {prompt}" if prompt else REFCONTROL_TRIGGER


def build_depth_refcontrol_edit_prompt(
    *,
    diffusion_model: str,
    text_encoder_model: str,
    vae_model: str,
    lora_model_name: str,
    reference_image_name: str,
    structure_image_name: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int = 20,
    cfg: float = 5.0,
    lora_strength: float = 1.0,
) -> dict[str, Any]:
    positive_prompt = _ensure_refcontrol_prompt(prompt)
    nodes: dict[str, Any] = {
        "1": _node("UNETLoader", unet_name=diffusion_model, weight_dtype="default"),
        "2": _node("CLIPLoader", clip_name=text_encoder_model, type="flux2", device="default"),
        "3": _node("VAELoader", vae_name=vae_model),
        "4": _node(LORA_LOADER_MODEL_ONLY_CLASS, model=_link("1"), lora_name=lora_model_name, strength_model=lora_strength),
        "5": _node("CLIPTextEncode", clip=_link("2"), text=positive_prompt),
        "6": _node("CLIPTextEncode", clip=_link("2"), text=negative),
        "7": _node("LoadImage", image=reference_image_name),
        "8": _node(
            "ImageScaleToTotalPixels",
            image=_link("7"),
            upscale_method="nearest-exact",
            megapixels=1.0,
            resolution_steps=1,
        ),
        "9": _node("VAEEncode", pixels=_link("8"), vae=_link("3")),
        "10": _node(REFERENCE_LATENT_CLASS, conditioning=_link("5"), latent=_link("9")),
        "11": _node(REFERENCE_LATENT_CLASS, conditioning=_link("6"), latent=_link("9")),
        "12": _node("LoadImage", image=structure_image_name),
        "13": _node(DEPTH_ANYTHING_V2_PREPROCESSOR_CLASS, image=_link("12"), ckpt_name="depth_anything_v2_vitl.pth", resolution=512),
        "14": _node(
            "ImageScaleToTotalPixels",
            image=_link("13"),
            upscale_method="nearest-exact",
            megapixels=1.0,
            resolution_steps=1,
        ),
        "15": _node("VAEEncode", pixels=_link("14"), vae=_link("3")),
        "16": _node(REFERENCE_LATENT_CLASS, conditioning=_link("10"), latent=_link("15")),
        "17": _node(REFERENCE_LATENT_CLASS, conditioning=_link("11"), latent=_link("15")),
        "18": _node("Flux2Scheduler", steps=steps, width=1024, height=1024),
        "19": _node("KSamplerSelect", sampler_name="euler"),
        "20": _node("RandomNoise", noise_seed=seed),
        "21": _node("CFGGuider", model=_link("4"), positive=_link("16"), negative=_link("17"), cfg=cfg),
        "22": _node(
            "SamplerCustomAdvanced",
            noise=_link("20"),
            guider=_link("21"),
            sampler=_link("19"),
            sigmas=_link("18"),
            latent_image=_link("15"),
        ),
        "23": _node("VAEDecode", samples=_link("22"), vae=_link("3")),
        "24": _node("SaveImage", images=_link("23"), filename_prefix="Flux2-Klein-RefControl-Depth"),
    }
    return nodes


def build_rtx_upscale_prompt(
    *,
    image: str,
    resize_type: str = "scale by multiplier",
    scale: float = 2.0,
    width: int | None = None,
    height: int | None = None,
    quality: str = "ULTRA",
    filename_prefix: str = "Upscaled",
) -> dict[str, Any]:
    inputs: dict[str, Any] = {"images": _link("1"), "resize_type": resize_type, "quality": quality}
    if resize_type == "target dimensions":
        if width is None or height is None:
            raise ValueError("width and height are required when resize_type is 'target dimensions'.")
        inputs["width"] = width
        inputs["height"] = height
    else:
        inputs["scale"] = scale
    return {
        "1": _node("LoadImage", image=image),
        "2": _node(RTX_VIDEO_SUPER_RESOLUTION_CLASS, **inputs),
        "3": _node("SaveImage", images=_link("2"), filename_prefix=filename_prefix),
    }


def build_esrgan_upscale_prompt(
    *,
    image: str,
    upscale_model_name: str = DEFAULT_UPSCALE_MODEL,
    resize_type: str = "scale by multiplier",
    scale: float = 2.0,
    target_width: int | None = None,
    target_height: int | None = None,
    filename_prefix: str = "Upscaled",
) -> dict[str, Any]:
    nodes: dict[str, Any] = {
        "1": _node("LoadImage", image=image),
        "2": _node(UPSCALE_MODEL_LOADER_CLASS, model_name=upscale_model_name),
        "3": _node(IMAGE_UPSCALE_WITH_MODEL_CLASS, upscale_model=_link("2"), image=_link("1")),
    }
    output_link = _link("3")
    if resize_type == "target dimensions":
        if target_width is None or target_height is None:
            raise ValueError("target_width and target_height are required when resize_type is 'target dimensions'.")
        nodes["4"] = _node(
            "ImageScaleToTotalPixels",
            image=_link("3"),
            upscale_method="lanczos",
            megapixels=(target_width * target_height) / 1_048_576,
            resolution_steps=1,
        )
        output_link = _link("4")
    elif scale != 2.0:
        # Keep the API symmetrical with the RTX path; the Real-ESRGAN model itself is 2x.
        pass
    nodes[str(len(nodes) + 1)] = _node("SaveImage", images=output_link, filename_prefix=filename_prefix)
    return nodes


def dump_workflow_templates(output_dir: Path | None = None) -> tuple[Path, Path]:
    target_dir = output_dir or WORKFLOWS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    edit = build_edit_prompt(
        diffusion_model="flux-2-klein-4b-fp8.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="turn this image into a realistic photo",
        negative="anime, cartoon, low quality",
        seed=0,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )
    t2i = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b-fp8.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="a cinematic portrait photo",
        negative="blurry, cartoon, low quality",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )
    edit_path = target_dir / "flux2_klein_edit.json"
    t2i_path = target_dir / "flux2_klein_t2i.json"
    edit_path.write_text(json.dumps(edit, indent=2), encoding="utf-8")
    t2i_path.write_text(json.dumps(t2i, indent=2), encoding="utf-8")
    return edit_path, t2i_path
