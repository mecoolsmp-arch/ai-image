from __future__ import annotations

from pathlib import Path

import pytest

from comfyui_app import installer


@pytest.mark.parametrize(
    ("torch_runtime", "expected_suffix"),
    [
        (("2.6.0", 2, 6, "12.6"), "sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl"),
        (("2.7.1", 2, 7, "12.8"), "sageattention-2.2.0+cu128torch2.7.1.post3-cp39-abi3-win_amd64.whl"),
        (("2.8.0", 2, 8, "12.8"), "sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"),
        (("2.9.0", 2, 9, "12.8"), "sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"),
        (("2.9.0", 2, 9, "13.0"), "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"),
        (("2.10.0", 2, 10, "12.8"), "sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"),
        (("2.10.0", 2, 10, "13.0"), "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"),
    ],
)
def test_sageattention_wheel_selector_matches_supported_torch_cuda_combinations(monkeypatch, torch_runtime, expected_suffix) -> None:
    monkeypatch.setattr(installer, "_torch_runtime", lambda: torch_runtime)

    wheel_url = installer._sageattention_wheel_url()

    assert wheel_url is not None
    assert wheel_url.endswith(expected_suffix)


def test_sageattention_wheel_selector_falls_back_when_no_match(monkeypatch) -> None:
    monkeypatch.setattr(installer, "_torch_runtime", lambda: ("2.5.1", 2, 5, "12.4"))

    assert installer._sageattention_wheel_url() is None


def test_patch_teacache_lightricks_import_rewrites_and_is_idempotent(tmp_path: Path) -> None:
    nodes_py = tmp_path / "nodes.py"
    original_text = (
        "from comfy.ldm.lightricks.model import precompute_freqs_cis\n"
        "print('tea')\n"
    )
    nodes_py.write_text(original_text, encoding="utf-8")

    patched = installer._patch_teacache_lightricks_import(nodes_py)
    first_pass = nodes_py.read_text(encoding="utf-8")
    second = installer._patch_teacache_lightricks_import(nodes_py)
    second_pass = nodes_py.read_text(encoding="utf-8")

    expected_block = (
        "try:\n"
        "    from comfy.ldm.lightricks.model import precompute_freqs_cis\n"
        "except ImportError:\n"
        "    try:\n"
        "        from comfy.ldm.lightricks.model import LTXBaseModel\n"
        "        precompute_freqs_cis = LTXBaseModel.precompute_freqs_cis\n"
        "    except (ImportError, AttributeError):\n"
        "        def precompute_freqs_cis(*args, **kwargs):\n"
        "            raise RuntimeError(\"TeaCache LTX-Video support needs precompute_freqs_cis, which is unavailable in this ComfyUI version.\")\n"
    )

    assert patched is True
    assert expected_block in first_pass
    assert second is False
    assert second_pass == first_pass


def test_patch_teacache_lightricks_import_leaves_unmatched_content_unchanged(tmp_path: Path) -> None:
    nodes_py = tmp_path / "nodes.py"
    original_text = "print('no lightricks import here')\n"
    nodes_py.write_text(original_text, encoding="utf-8")

    patched = installer._patch_teacache_lightricks_import(nodes_py)

    assert patched is False
    assert nodes_py.read_text(encoding="utf-8") == original_text


def test_patch_teacache_flux_forward_signature_rewrites_and_is_idempotent(tmp_path: Path) -> None:
    nodes_py = tmp_path / "nodes.py"
    original_text = (
        "def teacache_flux_forward(\n"
        "    self,\n"
        "    img: Tensor,\n"
        "    img_ids: Tensor,\n"
        "    txt: Tensor,\n"
        "    txt_ids: Tensor,\n"
        "    timesteps: Tensor,\n"
        "    y: Tensor,\n"
        "    guidance: Tensor = None,\n"
        "    control = None,\n"
        "    transformer_options={},\n"
        "    attn_mask: Tensor = None,\n"
        "    ) -> Tensor:\n"
        "    return img\n"
    )
    nodes_py.write_text(original_text, encoding="utf-8")

    patched = installer._patch_teacache_flux_forward_signature(nodes_py)
    first_pass = nodes_py.read_text(encoding="utf-8")
    second = installer._patch_teacache_flux_forward_signature(nodes_py)
    second_pass = nodes_py.read_text(encoding="utf-8")

    assert patched is True
    assert "def teacache_flux_forward(" in first_pass
    assert "timestep_zero_index=None" in first_pass
    assert "**kwargs" in first_pass
    assert second is False
    assert second_pass == first_pass


def test_patch_teacache_flux_forward_signature_leaves_unmatched_content_unchanged(tmp_path: Path) -> None:
    nodes_py = tmp_path / "nodes.py"
    original_text = "print('no flux forward here')\n"
    nodes_py.write_text(original_text, encoding="utf-8")

    patched = installer._patch_teacache_flux_forward_signature(nodes_py)

    assert patched is False
    assert nodes_py.read_text(encoding="utf-8") == original_text
