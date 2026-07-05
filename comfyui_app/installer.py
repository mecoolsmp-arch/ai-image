from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from comfyui_app.config import COMFYUI_DIR, DOTENV_PATH, REPO_ROOT, get_hf_token
from comfyui_app.model_resolver import (
    ModelResolverError,
    download_models,
    resolve_consistency_lora_models,
    resolve_depth_control_models,
    resolve_models,
)
from comfyui_app.vram import detect_vram, select_tier

logger = logging.getLogger(__name__)


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        detail = stderr or stdout or "no output"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")


def _git_clone_or_pull(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        _run(["git", "-C", str(target_dir), "pull"])
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(target_dir)])


def _install_requirements(requirements_file: Path) -> None:
    if requirements_file.exists():
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])


def _torch_runtime() -> tuple[str, int, int, str]:
    import torch

    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not match:
        raise RuntimeError(f"Unable to parse the installed torch version: {torch.__version__}")
    cuda_version = str(torch.version.cuda or "")
    return torch.__version__, int(match.group(1)), int(match.group(2)), cuda_version


def _sageattention_wheel_url() -> str | None:
    _, major, minor, cuda_version = _torch_runtime()
    if major != 2:
        return None
    if minor == 6 and cuda_version.startswith("12.6"):
        return "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu126torch2.6.0.post3-cp39-abi3-win_amd64.whl"
    if minor == 7 and cuda_version.startswith("12."):
        return "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.7.1.post3-cp39-abi3-win_amd64.whl"
    if minor == 8 and cuda_version.startswith("12."):
        return "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"
    if minor >= 9 and (cuda_version.startswith("12.") or cuda_version.startswith("13.")):
        wheel = "sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" if cuda_version.startswith("13.") else "sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
        release = "v2.2.0-windows.post4"
        return f"https://github.com/woct0rdho/SageAttention/releases/download/{release}/{wheel}"
    return None


def _install_sageattention() -> None:
    print("Trying to install SageAttention for the current torch/CUDA combination...")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "triton-windows"])
    torch_version, _, _, cuda_version = _torch_runtime()
    print(f"Detected torch {torch_version} with CUDA {cuda_version or 'unknown'}.")
    wheel_url = _sageattention_wheel_url()
    if wheel_url is None:
        print("No matching SageAttention wheel was found for this torch/CUDA pair; falling back to pip install sageattention.")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "sageattention"])
        print("SageAttention fallback install finished. Relaunch with Launch.bat.")
        return
    try:
        print(f"Installing SageAttention wheel: {wheel_url.rsplit('/', 1)[-1]}")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", wheel_url])
    except Exception as exc:
        print(f"WARNING: The SageAttention wheel install did not finish: {exc}")
        print("Falling back to the PyPI package.")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "sageattention"])
    print("SageAttention install finished. On an RTX 3070, legacy kernels are expected; relaunch with Launch.bat.")



def _install_rtx_vsr_support() -> None:
    print("Trying to add the RTX video super-resolution node...")
    try:
        custom_nodes = COMFYUI_DIR / "custom_nodes"
        custom_nodes.mkdir(parents=True, exist_ok=True)
        _install_custom_node(
            "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git",
            custom_nodes / "Nvidia_RTX_Nodes_ComfyUI",
        )
    except Exception as exc:
        print(f"WARNING: The RTX VSR custom node could not be prepared: {exc}")
        print("The app will fall back to Real-ESRGAN for upscaling.")

    try:
        print("Trying to install the nvidia-vfx package...")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "nvidia-vfx"])
    except Exception as exc:
        print(f"WARNING: The nvidia-vfx install did not finish: {exc}")
        print("RTX video super-resolution will not be available, so the app will use Real-ESRGAN.")


def _install_depth_control_support() -> None:
    print("Trying to add the depth pose/shape lock assets...")
    try:
        custom_nodes = COMFYUI_DIR / "custom_nodes"
        custom_nodes.mkdir(parents=True, exist_ok=True)
        _install_custom_node(
            "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
            custom_nodes / "comfyui_controlnet_aux",
        )
    except Exception as exc:
        print(f"WARNING: The ControlNet Aux custom node could not be prepared: {exc}")
        print("Pose/Shape lock (depth) will not be available.")


def _refresh_models() -> None:
    token = _get_token_from_user()
    vram_gb, device_name, cuda_available = detect_vram()
    if not cuda_available or vram_gb <= 0.0:
        raise ModelResolverError("No NVIDIA CUDA GPU was detected, so ComfyUI cannot be prepared on this machine.")
    tier = select_tier(vram_gb)
    print(f"Detected {device_name} with about {vram_gb:.1f} GB of VRAM.")
    print(f"Using the {tier.label} setup.")
    if vram_gb < 7.0:
        print("This GPU is on the low-memory path, so ComfyUI will use the lighter model choices.")
    resolved = resolve_models(vram_gb, token)
    download_models(resolved, token, progress_cb=print)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the ComfyUI stack for the local image app.")
    parser.add_argument("--refresh-models", action="store_true", help="Only refresh the model resolver step.")
    parser.add_argument(
        "--with-experimental-speedups",
        action="store_true",
        help="Also try the optional Nunchaku, SageAttention, Triton, speed paths.",
    )
    parser.add_argument(
        "--with-depth-control",
        action="store_true",
        help="Also install the optional FLUX.2 depth pose/shape lock assets.",
    )
    parser.add_argument(
        "--with-consistency-lora",
        action="store_true",
        help="Also download the optional FLUX.2 Klein consistency LoRA asset.",
    )
    parser.add_argument(
        "--install-sageattention",
        action="store_true",
        help="Install SageAttention in the active venv and exit.",
    )
    args = parser.parse_args(argv)

    try:
        if args.install_sageattention:
            _install_sageattention()
            return 0
        if not args.refresh_models:
            _git_clone_or_pull("https://github.com/comfyanonymous/ComfyUI.git", COMFYUI_DIR)
            _install_requirements(COMFYUI_DIR / "requirements.txt")
            custom_nodes = COMFYUI_DIR / "custom_nodes"
            custom_nodes.mkdir(parents=True, exist_ok=True)
            _install_custom_node("https://github.com/city96/ComfyUI-GGUF.git", custom_nodes / "ComfyUI-GGUF")
            _install_custom_node("https://github.com/ltdrdata/ComfyUI-Manager.git", custom_nodes / "ComfyUI-Manager")
            _install_rtx_vsr_support()
            if args.with_experimental_speedups:
                _install_experimental_speedups()
        _refresh_models()
        if args.with_depth_control:
            token = _get_token_from_user()
            depth_resolved = resolve_depth_control_models(token)
            download_models(depth_resolved, token, progress_cb=print)
            _install_depth_control_support()
        if args.with_consistency_lora:
            token = _get_token_from_user()
            consistency_resolved = resolve_consistency_lora_models(token)
            download_models(consistency_resolved, token, progress_cb=print)
    except ModelResolverError as exc:
        print(exc.message)
        return 2
    except RuntimeError as exc:
        print(f"Setup failed: {exc}")
        return 3
    except subprocess.CalledProcessError as exc:
        print(f"A git or pip command failed: {exc}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
