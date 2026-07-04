from __future__ import annotations

from pathlib import Path

from comfyui_app import model_resolver
from comfyui_app.model_resolver import resolve_models
from comfyui_app.vram import select_tier
from comfyui_app.workflow_builder import build_edit_prompt, build_t2i_prompt


def _tree(*paths: str, size: int = 1) -> list[dict[str, object]]:
    return [{"path": path, "size": size, "oid": f"oid-{index}"} for index, path in enumerate(paths, start=1)]


def test_select_tier_for_rtx_3070_8gb() -> None:
    tier = select_tier(8.0)
    assert tier.diffusion == "flux2_fp8"
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
        diffusion_model="flux-2-klein-4b-fp8.safetensors",
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
    assert prompt["13"]["inputs"]["steps"] == 4
    assert prompt["15"]["inputs"]["noise_seed"] == 123


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
        diffusion_model="flux-2-klein-4b-fp8.safetensors",
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


def test_build_t2i_prompt_omits_image_encoding_nodes() -> None:
    prompt = build_t2i_prompt(
        diffusion_model="flux-2-klein-4b-fp8.safetensors",
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
    assert "EmptyFlux2LatentImage" in class_types
    assert "SamplerCustomAdvanced" in class_types


def test_resolver_regex_match_picks_fp8(monkeypatch) -> None:
    trees = {
        "black-forest-labs/FLUX.2-klein-4b-fp8": _tree(
            "flux-2-klein-4b-fp8.safetensors",
            "notes.txt",
        ),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token")

    assert Path(str(resolved["diffusion"]["local_filename"])).name == "flux-2-klein-4b-fp8.safetensors"
    assert Path(str(resolved["text_encoder"]["local_filename"])).name == "qwen_3_4b_fp4_flux2.safetensors"
    assert Path(str(resolved["vae"]["local_filename"])).name == "flux2-vae.safetensors"


def test_resolver_prefers_tiny_decoder_when_present(monkeypatch) -> None:
    trees = {
        "black-forest-labs/FLUX.2-klein-4b-fp8": _tree("flux-2-klein-4b-fp8.safetensors"),
        "Comfy-Org/flux2-klein-4B": _tree(
            "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
            "split_files/vae/flux2-vae.safetensors",
            "split_files/vae/flux2-taef2.safetensors",
        ),
    }

    def fake_fetch(repo: str, token: str | None):
        return trees.get(repo, [])

    monkeypatch.setattr(model_resolver, "_fetch_repo_tree", fake_fetch)
    resolved = resolve_models(8.0, token="token")

    assert "taef2" in Path(str(resolved["vae"]["local_filename"])).name.lower()
