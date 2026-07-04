"""Startup preflight checks: dependency verification, CUDA 13 enforcement,
and optional dependency probing.

Extracted from app.py to keep the main entry point lean.
"""

import importlib
import importlib.util
import json
import os
from typing import List, Tuple

import torch

from src.config import BASE_DIR, DEPENDENCY_CHECK_FLAG
from src.runtime_policies import (
    build_dependency_profile_metadata,
    compute_file_sha256,
    describe_acceleration_stack,
    is_cuda13_runtime,
    is_dependency_metadata_current,
    select_requirements_file,
)


# ---------------------------------------------------------------------------
# Module availability helpers
# ---------------------------------------------------------------------------

def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _probe_optional_dependency_statuses() -> List[Tuple[str, str, str]]:
    """Probe optional packages and return (level, name, detail) tuples."""
    statuses: List[Tuple[str, str, str]] = []

    for import_name, display_name in [
        ("triton", "Triton"),
        ("sageattention", "SageAttention"),
        ("hf_xet", "HF Xet"),
    ]:
        if _module_available(import_name):
            statuses.append(("OK", display_name, "available"))
        else:
            statuses.append(("WARN", display_name, "package not installed"))

    return statuses


# ---------------------------------------------------------------------------
# Core startup checks
# ---------------------------------------------------------------------------

def check_required_dependencies() -> bool:
    """Verify critical imports without mutating the environment."""
    print("Checking required dependencies...")

    required_checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("accelerate", "Accelerate"),
        ("safetensors", "SafeTensors"),
        ("sentencepiece", "SentencePiece"),
        ("google.protobuf", "Protobuf"),
        ("PIL", "Pillow"),
        ("gradio", "Gradio"),
        ("scipy", "SciPy"),
        ("peft", "PEFT"),
        ("optimum.quanto", "Optimum Quanto"),
        ("requests", "Requests"),
    ]

    failed_imports = []
    for import_name, display_name in required_checks:
        try:
            if "." in import_name:
                importlib.import_module(import_name)
            else:
                __import__(import_name)
            print(f"[OK] {display_name} available")
        except Exception as exc:
            print(f"[FAIL] {display_name}: {exc}")
            failed_imports.append((import_name, str(exc)))

    for level, display_name, detail in _probe_optional_dependency_statuses():
        print(f"[{level}] {display_name} {detail}")

    if not failed_imports:
        print("All required dependencies are available.")
        return True

    print(f"\nDependency verification failed for {len(failed_imports)} module(s).")
    print("Run Install.bat --repair, then relaunch.")
    return False


def _load_dependency_metadata(flag_path: str) -> dict:
    if not os.path.exists(flag_path):
        return {}
    try:
        with open(flag_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        try:
            with open(flag_path, "r", encoding="utf-8") as handle:
                raw_value = handle.read().strip()
            if raw_value:
                return {"_legacy_metadata_text": raw_value}
        except Exception:
            pass
        return {}


def run_dependency_preflight() -> bool:
    """Verify dependency profile and imports without mutating the environment."""
    requirements_file = select_requirements_file(
        base_dir=BASE_DIR,
        is_windows=(os.name == "nt"),
        cuda_available=torch.cuda.is_available(),
    )
    requirements_path = os.path.join(BASE_DIR, requirements_file)
    requirements_hash = compute_file_sha256(requirements_path)
    target_metadata = build_dependency_profile_metadata(requirements_file, requirements_hash)
    current_metadata = _load_dependency_metadata(DEPENDENCY_CHECK_FLAG)
    is_current = is_dependency_metadata_current(current_metadata, target_metadata)

    if not os.path.exists(requirements_path):
        print(f"Dependency preflight failed: missing requirements file {requirements_path}")
        return False

    if not is_current:
        if current_metadata.get("_legacy_metadata_text"):
            print("Dependency metadata is in legacy format and must be regenerated.")
        print(f"Dependency profile is out of date for {requirements_file}.")
        print("Run Install.bat --repair, then relaunch.")
        return False

    print("Dependency profile is current.")
    if not check_required_dependencies():
        print("Dependency preflight failed during import verification.")
        print("Run Install.bat --repair, then relaunch.")
        return False

    return True


def enforce_cuda13_runtime_profile() -> bool:
    """Enforce CUDA 13 runtime profile on Windows when CUDA is available."""
    if os.name != "nt":
        return True
    if os.environ.get("UFIG_ENFORCE_CUDA13", "1") != "1":
        return True
    if not torch.cuda.is_available():
        print("CUDA 13 profile check failed: CUDA device not available.")
        print("This workflow is configured for CUDA 13 + NVIDIA GPU.")
        print("Re-run Install.bat --repair and verify NVIDIA drivers/runtime.")
        return False

    cuda_runtime = getattr(torch.version, "cuda", None)
    if is_cuda13_runtime(cuda_runtime):
        print(f"CUDA runtime profile validated: {cuda_runtime}")
        return True

    print(f"CUDA 13 profile check failed: detected torch CUDA runtime '{cuda_runtime}'.")
    print("This project requires CUDA 13 wheels. Re-run Install.bat --repair.")
    return False


def ensure_accelerate_available() -> str:
    """Ensure accelerate is importable; raise RuntimeError otherwise."""
    try:
        import accelerate
        return accelerate.__version__
    except Exception as exc:
        raise RuntimeError(
            "Accelerate is required to load SDNQ quantized models with low_cpu_mem_usage=True. "
            "Re-run Launch.bat to reinstall dependencies or install with `pip install accelerate`."
        ) from exc


def log_provider_telemetry(
    profile: str,
    cuda_speed_status: dict,
    compile_probe: bool,
    optional_accelerators: bool,
) -> None:
    """Log startup provider telemetry and fallback context."""
    runtime_stack = {
        "profile": profile,
        "cuda_runtime": getattr(torch.version, "cuda", None) or "none",
        "tf32": cuda_speed_status.get("tf32", False),
        "matmul_precision": cuda_speed_status.get("matmul_precision") or "default",
        "sdp": cuda_speed_status.get("sdp", False),
        "allocator_conf": cuda_speed_status.get("allocator_conf"),
        "compile_probe": compile_probe,
        "optional_accelerators": optional_accelerators,
    }
    try:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Runtime] GPU: {torch.cuda.get_device_name(0)} ({total_gb:.2f} GB)")
        else:
            print("[Runtime] GPU: CUDA unavailable")
    except Exception as e:
        print(f"[Runtime] GPU telemetry unavailable: {e}")

    print(f"[Runtime] Acceleration stack: {describe_acceleration_stack(runtime_stack)}")


def ensure_cache_dirs() -> None:
    """Create cache directories from environment variables."""
    for path in [
        os.environ.get("UFIG_CACHE_DIR", os.path.join(BASE_DIR, "cache")),
        os.environ.get("HF_HOME", ""),
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HF_XET_CACHE", ""),
        os.environ.get("HF_ASSETS_CACHE", ""),
        os.environ.get("TORCH_HOME", ""),
        os.environ.get("GRADIO_TEMP_DIR", ""),
    ]:
        if path:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                pass
