from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from comfyui_app.config import COMFYUI_DIR, DOTENV_PATH, REPO_ROOT, get_hf_token
from comfyui_app.model_resolver import ModelResolverError, download_models, resolve_depth_control_models, resolve_models
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


def _load_env_lines(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _save_token(token: str) -> None:
    existing = _load_env_lines(DOTENV_PATH)
    existing["HF_TOKEN"] = token
    payload = "\n".join(f"{key}={value}" for key, value in existing.items()) + "\n"
    DOTENV_PATH.write_text(payload, encoding="utf-8")
    os.environ["HF_TOKEN"] = token


def _get_token_from_user() -> str:
    token = get_hf_token()
    if token:
        return token
    print("A Hugging Face token is needed to download the model files.")
    token = input("Paste your Hugging Face token and press Enter: ").strip()
    if not token:
        raise ModelResolverError("No Hugging Face token was provided.")
    _save_token(token)
    return token


def _install_custom_node(repo_url: str, target_dir: Path) -> None:
    _git_clone_or_pull(repo_url, target_dir)
    requirements_file = target_dir / "requirements.txt"
    if requirements_file.exists():
        _install_requirements(requirements_file)


def _install_experimental_speedups() -> None:
    print("Experimental speedups are enabled. This step is optional and best effort.")
    try:
        custom_nodes = COMFYUI_DIR / "custom_nodes"
        custom_nodes.mkdir(parents=True, exist_ok=True)
        _install_custom_node("https://github.com/nunchaku-ai/ComfyUI-nunchaku.git", custom_nodes / "ComfyUI-nunchaku")
    except Exception as exc:
        print(f"WARNING: The Nunchaku custom node could not be prepared: {exc}")
        print("You can still use the default FP8 workflow.")

    try:
        print("Trying to install the Nunchaku Python package...")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "nunchaku"])
    except Exception as exc:
        print(f"WARNING: The Nunchaku package install did not finish: {exc}")
        print("If you want to try the experimental path later, follow the Nunchaku releases page.")

    try:
        print("Trying to install the Windows Triton build for torch.compile...")
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "triton-windows"])
    except Exception as exc:
        print(f"WARNING: The Triton install did not finish: {exc}")
        print("torch.compile requires Triton on Windows, so it will otherwise error.")


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
        help="Also try the optional Nunchaku experimental speed path.",
    )
    parser.add_argument(
        "--with-depth-control",
        action="store_true",
        help="Also install the optional FLUX.2 depth pose/shape lock assets.",
    )
    args = parser.parse_args(argv)

    try:
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
