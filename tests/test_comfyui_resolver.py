from __future__ import annotations

from pathlib import Path

from comfyui_app import model_resolver
from comfyui_app import workflow_builder
from comfyui_app.model_resolver import ModelResolverError, resolve_models
from comfyui_app.vram import select_tier
from comfyui_app.workflow_builder import (
    build_depth_refcontrol_edit_prompt,
    build_edit_prompt,
    build_mrflow_edit_prompt,
    build_mrflow_t2i_prompt,
    build_t2i_prompt,
)


def _tree(*paths: str, size: int = 1) -> list[dict[str, object]]:
    return [{"path": path, "size": size, "oid": f"oid-{index}"} for index, path in enumerate(paths, start=1)]


def test_select_tier_for_rtx_3070_8gb() -> None:
    tier = select_tier(8.0)
    assert tier.diffusion == "int8"
    assert tier.text_encoder == "flux2_fp4"
    assert tier.use_tiled_decode is True
    assert tier.extra_launch_flags == []


def test_select_tier_for_6gb_uses_gguf() -> None:
    tier = select_tier(6.0)
    assert tier.diffusion == "flux2_gguf_q4_k_m"
    assert tier.text_encoder == "flux2_fp4"
    assert tier.use_tiled_decode is True
    assert tier.extra_launch_flags == ["--lowvram"]


def test_build_edit_prompt_injects_expected_values() -> None:
    prompt = build_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    class_types = {node["class_type"] for node in prompt.values()}
    assert {"UNETLoader", "CLIPLoader", "VAELoader", "Flux2Scheduler", "SamplerCustomAdvanced", "ReferenceLatent", "VAEEncode", "VAEDecodeTiled", "SaveImage"} <= class_types
    assert prompt["4"]["inputs"]["text"] == "make this photo realistic"
    assert prompt["5"]["inputs"]["text"] == "blurry, cartoon"
    assert prompt["7"]["inputs"]["megapixels"] == 1.0
    assert prompt["7"]["inputs"]["resolution_steps"] == 1
    assert prompt["13"]["inputs"]["steps"] == 4
    assert prompt["15"]["inputs"]["noise_seed"] == 123
    assert prompt["18"]["inputs"]["overlap"] == 64
    assert prompt["18"]["inputs"]["temporal_size"] == 64
    assert prompt["18"]["inputs"]["temporal_overlap"] == 8


def test_build_edit_prompt_switches_loader_for_gguf() -> None:
    gguf_prompt = build_edit_prompt(
        diffusion_model="flux-2-klein-4b-Q4_K_M.gguf",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )
    safetensors_prompt = build_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    assert gguf_prompt["1"]["class_type"] == "UnetLoaderGGUF"
    assert safetensors_prompt["1"]["class_type"] == "UNETLoader"


def test_build_edit_prompt_adds_consistency_lora_and_rewires_cfg() -> None:
    prompt = build_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
        consistency_lora_name="my_consistency.safetensors",
        consistency_lora_strength=0.8,
    )

    lora_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "LoraLoaderModelOnly")
    assert prompt[lora_node_id]["inputs"]["model"] == ["1", 0]
    assert prompt[lora_node_id]["inputs"]["lora_name"] == "my_consistency.safetensors"
    assert prompt[lora_node_id]["inputs"]["strength_model"] == 0.8
    assert prompt["16"]["inputs"]["model"] == [lora_node_id, 0]


def test_build_edit_prompt_keeps_base_model_without_consistency_lora() -> None:
    prompt = build_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    assert all(node["class_type"] != "LoraLoaderModelOnly" for node in prompt.values())
    assert prompt["16"]["inputs"]["model"] == ["1", 0]


def test_build_edit_prompt_uses_nunchaku_loader_for_experimental_engine() -> None:
    prompt = build_edit_prompt(
        diffusion_model="svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="make this photo realistic",
        negative="blurry, cartoon",
        seed=123,
        steps=4,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
        engine="nunchaku_int4",
    )

    assert prompt["1"]["class_type"] == "NunchakuFluxDiTLoader"
    assert prompt["1"]["inputs"]["model_path"] == "svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors"
    assert prompt["16"]["inputs"]["model"] == ["1", 0]


def test_build_t2i_prompt_omits_image_encoding_nodes() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
    )

    class_types = {node["class_type"] for node in prompt.values()}
    assert "LoadImage" not in class_types
    assert "VAEEncode" not in class_types


def test_build_t2i_prompt_keeps_base_model_without_consistency_lora() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
    )

    assert all(node["class_type"] != "LoraLoaderModelOnly" for node in prompt.values())
    assert prompt["10"]["inputs"]["model"] == ["1", 0]


def test_build_t2i_prompt_adds_consistency_lora_and_rewires_cfg() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
        consistency_lora_name="my_consistency.safetensors",
        consistency_lora_strength=0.8,
    )

    lora_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "LoraLoaderModelOnly")
    assert prompt[lora_node_id]["inputs"]["model"] == ["1", 0]
    assert prompt[lora_node_id]["inputs"]["lora_name"] == "my_consistency.safetensors"
    assert prompt[lora_node_id]["inputs"]["strength_model"] == 0.8
    assert prompt["10"]["inputs"]["model"] == [lora_node_id, 0]


def test_build_t2i_prompt_composes_consistency_lora_with_torch_compile() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
        consistency_lora_name="my_consistency.safetensors",
        consistency_lora_strength=0.8,
        use_torch_compile=True,
    )

    lora_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "LoraLoaderModelOnly")
    compile_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "TorchCompileModel")
    assert prompt[compile_node_id]["inputs"]["model"] == [lora_node_id, 0]
    assert prompt["10"]["inputs"]["model"] == [compile_node_id, 0]


def test_build_depth_refcontrol_edit_prompt_uses_depth_assets() -> None:
    prompt = build_depth_refcontrol_edit_prompt(
        diffusion_model="flux-2-klein-base-4b-Q8_0.gguf",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        lora_model_name="flux2_klein_4b_refcontrol_depth.safetensors",
        reference_image_name="reference.png",
        structure_image_name="structure.png",
        prompt="a character portrait",
        negative="blurry",
        seed=77,
    )

    class_types = {node["class_type"] for node in prompt.values()}
    assert {"UnetLoaderGGUF", "LoraLoaderModelOnly", "DepthAnythingV2Preprocessor", "Flux2Scheduler", "CFGGuider", "SamplerCustomAdvanced", "VAEDecode", "SaveImage"} <= class_types
    assert prompt["1"]["class_type"] == "UnetLoaderGGUF"
    assert prompt["1"]["inputs"]["unet_name"] == "flux-2-klein-base-4b-Q8_0.gguf"
    assert prompt["4"]["inputs"]["lora_name"] == "flux2_klein_4b_refcontrol_depth.safetensors"
    assert prompt["5"]["inputs"]["text"].startswith("refcontrol")
    assert prompt["13"]["class_type"] == "DepthAnythingV2Preprocessor"
    assert prompt["13"]["inputs"]["resolution"] == 512
    assert prompt["14"]["class_type"] == "SaveImage"
    assert prompt["16"]["class_type"] == "GetImageSize"
    assert prompt["20"]["class_type"] == "EmptyFlux2LatentImage"
    assert prompt["21"]["inputs"]["width"] == ["16", 0]
    assert prompt["21"]["inputs"]["height"] == ["16", 1]
    assert prompt["21"]["inputs"]["steps"] == 20
    assert prompt["24"]["inputs"]["cfg"] == 5.0
    assert "SamplerCustomAdvanced" in class_types


def test_build_depth_refcontrol_edit_prompt_uses_safetensors_loader_for_int8() -> None:
    prompt = build_depth_refcontrol_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        lora_model_name="flux2_klein_4b_refcontrol_depth.safetensors",
        reference_image_name="reference.png",
        structure_image_name="structure.png",
        prompt="a character portrait",
        negative="blurry",
        seed=77,
    )

    assert prompt["1"]["class_type"] == "UNETLoader"
    assert prompt["1"]["inputs"]["unet_name"] == "flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"
    assert prompt["1"]["inputs"]["weight_dtype"] == "default"


def test_build_depth_refcontrol_edit_prompt_adds_teacache_after_lora(monkeypatch) -> None:
    monkeypatch.setattr(workflow_builder, "_teacache_available", lambda: True)

    prompt = build_depth_refcontrol_edit_prompt(
        diffusion_model="flux-2-klein-base-4b-Q8_0.gguf",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="flux2-vae.safetensors",
        lora_model_name="flux2_klein_4b_refcontrol_depth.safetensors",
        reference_image_name="reference.png",
        structure_image_name="structure.png",
        prompt="a character portrait",
        negative="blurry",
        seed=77,
        use_teacache=True,
    )

    tea_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "TeaCache")
    assert prompt[tea_node_id]["inputs"]["model"] == ["4", 0]
    assert prompt["24"]["inputs"]["model"] == [tea_node_id, 0]


def test_resolve_depth_control_models_supports_int8_base(monkeypatch) -> None:
    trees = {
        "unsloth/FLUX.2-klein-base-4B-GGUF": _tree("flux-2-klein-base-4b-Q8_0.gguf"),
        "thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora": _tree("flux2_klein_4b_refcontrol_depth.safetensors"),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)

    int8 = model_resolver.resolve_depth_control_models("token")

    assert Path(str(int8["depth_control_base_int8"]["local_filename"])).name == "flux-2-klein-base-4b-Q8_0.gguf"
    assert Path(str(int8["depth_control_lora"]["local_filename"])).name == "flux2_klein_4b_refcontrol_depth.safetensors"


def test_resolve_consistency_lora_models_selects_known_asset(monkeypatch) -> None:
    trees = {
        "lrzjason/Consistance_Edit_Lora": _tree("f2k_4B_consist_20260314.safetensors"),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)

    resolved = model_resolver.resolve_consistency_lora_models("token")

    assert Path(str(resolved["consistency_lora"]["local_filename"])).name == "f2k_4B_consist_20260314.safetensors"


def test_build_t2i_prompt_sets_new_decode_inputs_when_tiled() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    decode_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "VAEDecodeTiled")
    assert prompt[decode_node_id]["inputs"]["tile_size"] == 1024
    assert prompt[decode_node_id]["inputs"]["overlap"] == 64
    assert prompt[decode_node_id]["inputs"]["temporal_size"] == 64
    assert prompt[decode_node_id]["inputs"]["temporal_overlap"] == 8


def test_build_t2i_prompt_adds_teacache_node_when_available(monkeypatch) -> None:
    monkeypatch.setattr(workflow_builder, "_teacache_available", lambda: True)

    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=20,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
        use_teacache=True,
    )

    tea_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "TeaCache")
    assert prompt[tea_node_id]["inputs"]["model"] == ["1", 0]
    assert prompt[tea_node_id]["inputs"]["model_type"] == "flux"
    assert {"model", "model_type", "rel_l1_thresh", "start_percent", "end_percent", "cache_device"} <= set(prompt[tea_node_id]["inputs"])
    assert prompt["10"]["inputs"]["model"] == [tea_node_id, 0]


def test_build_t2i_prompt_skips_teacache_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(workflow_builder, "_teacache_available", lambda: False)

    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=20,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
        use_teacache=True,
    )

    assert all(node["class_type"] != "TeaCache" for node in prompt.values())
    assert prompt["10"]["inputs"]["model"] == ["1", 0]


def test_build_t2i_prompt_adds_torch_compile_node_when_requested() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        steps=4,
        cfg=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        use_tiled_decode=False,
        decode_tile_size=1024,
        use_torch_compile=True,
    )

    class_types = {node["class_type"] for node in prompt.values()}
    assert "TorchCompileModel" in class_types
    compile_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "TorchCompileModel")
    assert prompt[compile_node_id]["inputs"]["model"] == ["1", 0]
    assert prompt["10"]["inputs"]["model"] == [compile_node_id, 0]


def test_build_mrflow_t2i_prompt_adds_upscale_and_refine_nodes() -> None:
    prompt = build_mrflow_t2i_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        upscale_model_name="RealESRGAN_x2plus.pth",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        stage1_steps=4,
        refine_steps=1,
        refine_denoise=0.25,
        low_width=512,
        low_height=512,
        width=1024,
        height=1024,
        cfg=1.0,
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    assert prompt["13"]["class_type"] == "UpscaleModelLoader"
    assert prompt["14"]["class_type"] == "ImageUpscaleWithModel"
    assert prompt["14"]["inputs"]["upscale_model"] == ["13", 0]
    assert prompt["14"]["inputs"]["image"] == ["12", 0]
    assert prompt["17"]["class_type"] == "VAEEncode"
    assert prompt["17"]["inputs"]["pixels"] == ["15", 0]
    assert prompt["19"]["class_type"] == "SplitSigmasDenoise"
    assert prompt["23"]["class_type"] == "SamplerCustomAdvanced"
    assert prompt["23"]["inputs"]["latent_image"] == ["17", 0]
    assert prompt["23"]["inputs"]["sigmas"] == ["19", 1]


def test_build_mrflow_edit_prompt_adds_upscale_and_refine_nodes() -> None:
    prompt = build_mrflow_edit_prompt(
        diffusion_model="flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        text_encoder_model="qwen_3_4b_fp4_flux2.safetensors",
        vae_model="full_encoder_small_decoder.safetensors",
        upscale_model_name="RealESRGAN_x2plus.pth",
        prompt="a sunny portrait",
        negative="blurry",
        seed=0,
        stage1_steps=4,
        refine_steps=1,
        refine_denoise=0.25,
        low_width=512,
        low_height=512,
        width=0,
        height=0,
        cfg=1.0,
        megapixels=1.0,
        input_image_name="input.png",
        batch_size=1,
        use_tiled_decode=True,
        decode_tile_size=1024,
    )

    assert prompt["19"]["class_type"] == "UpscaleModelLoader"
    assert prompt["20"]["class_type"] == "ImageUpscaleWithModel"
    assert prompt["20"]["inputs"]["upscale_model"] == ["19", 0]
    assert prompt["20"]["inputs"]["image"] == ["18", 0]
    assert prompt["23"]["class_type"] == "VAEEncode"
    assert prompt["23"]["inputs"]["pixels"] == ["21", 0]
    assert prompt["25"]["class_type"] == "SplitSigmasDenoise"
    assert prompt["29"]["class_type"] == "SamplerCustomAdvanced"
    assert prompt["29"]["inputs"]["latent_image"] == ["23", 0]
    assert prompt["29"]["inputs"]["sigmas"] == ["25", 1]


def test_resolver_default_engine_picks_int8_diffusion(monkeypatch) -> None:
    trees = {
        "Bedovyy/FLUX.2-klein-4B-INT8-Comfy": _tree(
            "flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors",
        ),
        "black-forest-labs/FLUX.2-small-decoder": _tree(
            "full_encoder_small_decoder.safetensors",
        ),
        "2kpr/Real-ESRGAN": _tree("RealESRGAN_x2plus.pth"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token")

    assert Path(str(resolved["diffusion"]["local_filename"])).name == "flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"
    assert Path(str(resolved["text_encoder"]["local_filename"])).name == "qwen_3_4b_fp4_flux2.safetensors"
    assert Path(str(resolved["vae"]["local_filename"])).name == "full_encoder_small_decoder.safetensors"
    assert Path(str(resolved["upscale"]["local_filename"])).name == "RealESRGAN_x2plus.pth"


def test_resolver_prefers_gguf_when_requested(monkeypatch) -> None:
    trees = {
        "Bedovyy/FLUX.2-klein-4B-INT8-Comfy": _tree("flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"),
        "unsloth/FLUX.2-klein-4B-GGUF": _tree(
            "flux-2-klein-4b-Q4_K_M.gguf",
            "flux-2-klein-4b-Q3_K_M.gguf",
        ),
        "black-forest-labs/FLUX.2-small-decoder": _tree("full_encoder_small_decoder.safetensors"),
        "2kpr/Real-ESRGAN": _tree("RealESRGAN_x2plus.pth"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token", prefer_gguf=True)

    assert Path(str(resolved["diffusion"]["local_filename"])).name == "flux-2-klein-4b-Q4_K_M.gguf"


def test_resolver_engine_default_picks_int8_diffusion(monkeypatch) -> None:
    trees = {
        "Bedovyy/FLUX.2-klein-4B-INT8-Comfy": _tree("flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"),
        "black-forest-labs/FLUX.2-small-decoder": _tree("full_encoder_small_decoder.safetensors"),
        "2kpr/Real-ESRGAN": _tree("RealESRGAN_x2plus.pth"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token", engine="default")

    assert Path(str(resolved["diffusion"]["local_filename"])).name == "flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"


def test_resolver_engine_nunchaku_int4_picks_int4_diffusion(monkeypatch) -> None:
    trees = {
        "Bedovyy/FLUX.2-klein-4B-INT8-Comfy": _tree("flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"),
        "tonera/FLUX.2-klein-4B-Nunchaku": _tree("svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors"),
        "black-forest-labs/FLUX.2-small-decoder": _tree("full_encoder_small_decoder.safetensors"),
        "2kpr/Real-ESRGAN": _tree("RealESRGAN_x2plus.pth"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token", engine="nunchaku_int4")

    assert Path(str(resolved["diffusion"]["local_filename"])).name == "svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors"
    assert Path(str(resolved["vae"]["local_filename"])).name == "full_encoder_small_decoder.safetensors"
    assert Path(str(resolved["upscale"]["local_filename"])).name == "RealESRGAN_x2plus.pth"


def test_resolver_falls_back_to_flux2_vae_when_small_decoder_missing(monkeypatch) -> None:
    trees = {
        "Bedovyy/FLUX.2-klein-4B-INT8-Comfy": _tree("flux-2-klein-4b_learned_int8mixed_tensorwise.safetensors"),
        "2kpr/Real-ESRGAN": _tree("RealESRGAN_x2plus.pth"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        if repo == "black-forest-labs/FLUX.2-small-decoder":
            raise ModelResolverError("small decoder unavailable")
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token")

    assert Path(str(resolved["vae"]["local_filename"])).name == "flux2-vae.safetensors"
    assert Path(str(resolved["upscale"]["local_filename"])).name == "RealESRGAN_x2plus.pth"
