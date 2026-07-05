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
