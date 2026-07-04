import hashlib
import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple


OPTIMIZATION_PROFILES = ("max_speed", "balanced", "stability")
_TORCH_COMPILE_PROBE_STATUS: Dict[str, str] = {}
FAST_FLUX_MODEL_CHOICE = "FLUX.2-klein-4B (Int8)"
LOW_VRAM_FLUX_MODEL_CHOICE = "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"
FAST_RESOLUTION_PRESET = "~768px (Fast)"
STANDARD_RESOLUTION_PRESET = "~1024px"
BATCH_STANDARD_RESOLUTION_PRESET = "~1024px"


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_zimage_model(model_name: Optional[str]) -> bool:
    value = str(model_name or "").lower()
    return "z-image" in value or value.startswith("zimage")


def is_flux_model(model_name: Optional[str]) -> bool:
    value = str(model_name or "").lower()
    return "flux" in value or value.startswith("flux2-klein")


def should_show_edit_controls(model_name: Optional[str]) -> bool:
    return is_flux_model(model_name) or is_zimage_model(model_name)


def is_distilled_model(model_name: Optional[str]) -> bool:
    value = str(model_name or "").lower()
    return any(tag in value for tag in ["zimage", "z-image", "flux2-klein", "flux", "distilled"])


def is_sdnq_or_quantized(model_key: Optional[str], pipe: Optional[Any] = None) -> bool:
    value = str(model_key or "").lower()
    if any(tag in value for tag in ["sdnq", "zimage-quant", "int8", "quantized", "flux2-klein-int8"]):
        return True

    if pipe is None:
        return False

    try:
        pipe_type = str(type(pipe)).lower()
        if "sdnq" in pipe_type:
            return True
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            if "sdnq" in str(type(transformer)).lower():
                return True
            if "sdnq" in str(transformer).lower():
                return True
    except Exception:
        pass

    return False


def is_sdnq_model(model_key: Optional[str], pipe: Optional[Any] = None) -> bool:
    """True ONLY for SDNQ-quantized models (not optimum-quanto int8)."""
    value = str(model_key or "").lower()
    if "sdnq" in value:
        return True

    if pipe is None:
        return False

    try:
        pipe_type = str(type(pipe)).lower()
        if "sdnq" in pipe_type:
            return True
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            if "sdnq" in str(type(transformer)).lower():
                return True
            if "sdnq" in str(transformer).lower():
                return True
    except Exception:
        pass

    return False


def is_windows_3070_fast_profile(
    device: str,
    vram_gb: Optional[float] = None,
    gpu_name: Optional[str] = None,
    platform_system: Optional[str] = None,
) -> bool:
    if str(device).lower() != "cuda":
        return False

    system_name = str(platform_system or platform.system()).lower()
    if "win" not in system_name:
        return False

    normalized_gpu_name = str(gpu_name or "").lower()
    if normalized_gpu_name:
        if "3070" in normalized_gpu_name:
            return True
        return False

    if vram_gb is None:
        return False
    return 7.0 <= float(vram_gb) <= 8.6


def is_low_vram_ampere_fast_profile(
    device: str,
    vram_gb: Optional[float] = None,
    gpu_name: Optional[str] = None,
    platform_system: Optional[str] = None,
) -> bool:
    """
    OS-agnostic variant of `is_windows_3070_fast_profile`.

    Matches RTX 3070 / 3070 Ti (8 GB Ampere sm_86) on any host OS. The 3070
    is the reference card this project is tuned for, so the fast-path
    preset (~768px FLUX int8 default, VAE tiling, attention slicing, SDNQ
    + SDPA fallbacks) is applied on Linux/macOS as well as Windows.

    Note: We deliberately do NOT match the 12 GB RTX 3060, 16 GB RTX 4060
    Ti, 3050, or 4060 variants — those either have enough VRAM to run at
    1024px comfortably (3060 12 GB, 4060 Ti 16 GB) or have different
    architectures that need their own tuning.
    """
    if str(device).lower() != "cuda":
        return False

    normalized_gpu_name = str(gpu_name or "").lower()
    if normalized_gpu_name:
        # 3070 + 3070 Ti only. Both report "3070" in the product name and
        # both are 8 GB. No overlap with the 12 GB 3060 or 16 GB 4060 Ti.
        if "3070" in normalized_gpu_name:
            return True
        return is_windows_3070_fast_profile(
            device=device,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            platform_system=platform_system,
        )

    if vram_gb is None:
        return is_windows_3070_fast_profile(
            device=device,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            platform_system=platform_system,
        )
    # VRAM-only fallback (no name reported): match the 3070's ~8 GB band.
    # Kept tight (7.0-8.6 GB) to avoid picking up 6 GB laptop Ada cards or
    # the 12 GB 3060 (11.x GB reported after framebuffer reservation).
    return 7.0 <= float(vram_gb) <= 8.6


def resolve_default_flux_model_choice(
    device: str,
    vram_gb: Optional[float] = None,
    gpu_name: Optional[str] = None,
    platform_system: Optional[str] = None,
) -> str:
    if is_windows_3070_fast_profile(
        device=device,
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        platform_system=platform_system,
    ):
        return FAST_FLUX_MODEL_CHOICE
    return LOW_VRAM_FLUX_MODEL_CHOICE


def resolution_preset_to_long_edge(target_resolution: str) -> int:
    value = str(target_resolution or "")
    if "2048" in value or "2K" in value:
        return 2048
    if "1536" in value:
        return 1536
    if "1280" in value:
        return 1280
    if "768" in value:
        return 768
    return 1024


def resolve_default_resolution_preset(
    model_choice: Optional[str],
    mode: str = "single",
    device: str = "cuda",
    vram_gb: Optional[float] = None,
    gpu_name: Optional[str] = None,
    platform_system: Optional[str] = None,
) -> str:
    choice = str(model_choice or "")
    on_low_vram_ampere = is_low_vram_ampere_fast_profile(
        device=device,
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        platform_system=platform_system,
    )
    # Int8 FLUX path defaults to the aggressive ~768px preset on any low-VRAM
    # Ampere class GPU (Linux 3070 now included, not only Windows). The SDNQ
    # 4-bit path keeps the ~1024px default — its 4-bit weights already give
    # enough headroom that the extra resolution is worth the wall-clock.
    if "Int8" in choice and on_low_vram_ampere:
        return FAST_RESOLUTION_PRESET

    if mode == "batch":
        return BATCH_STANDARD_RESOLUTION_PRESET
    return STANDARD_RESOLUTION_PRESET


def is_klein_distilled_model(model_key: Optional[str]) -> bool:
    """True for FLUX.2 [klein] distilled variants (4-step inference)."""
    value = str(model_key or "").lower()
    if "klein" not in value:
        return False
    if "base" in value:
        return False
    return True


def resolve_default_inference_steps(
    model_key: Optional[str],
    requested_steps: Optional[int],
    optimization_profile: Optional[str] = None,
) -> int:
    """
    Clamp step count for distilled models whose quality plateaus at ~4 steps.

    - FLUX.2 [klein] distilled (4B/9B) — official recipe is 4 steps; going
      above ~8 is pure overhead with no quality gain.
    - Z-Image Turbo — 4-step model with identical behavior.
    - max_speed profile clamps to the official 4-step recipe for minimum
      latency; stability allows up to 8 steps for slightly better quality.
    - Everything else is passed through unchanged.
    """
    try:
        requested = int(requested_steps) if requested_steps is not None else 4
    except (TypeError, ValueError):
        requested = 4
    if requested <= 0:
        requested = 4

    if is_klein_distilled_model(model_key) or is_zimage_model(model_key):
        if str(optimization_profile or "").strip().lower() == "max_speed":
            return max(1, min(requested, 4))
        return max(1, min(requested, 8))
    return requested


def resolve_generation_guidance(model_key: Optional[str], guidance: float) -> float:
    final_guidance = 3.5 if guidance is None else float(guidance)
    # Z-Image Turbo models require guidance_scale=0.0 (no classifier-free
    # guidance). The official model card states: "Guidance should be 0 for
    # the Turbo models."
    if is_zimage_model(model_key):
        return 0.0
    # FLUX.2-klein-4B is distilled but still benefits from guidance.
    # The official example uses guidance_scale=1.0. Do NOT force 0.0.
    return final_guidance


def should_enable_autocast(
    device: str,
    model_key: Optional[str],
    pipe: Optional[Any] = None,
) -> bool:
    if str(device).lower() != "cuda":
        return False
    return not is_sdnq_or_quantized(model_key, pipe)


def canonicalize_cuda_allocator_conf(
    default_conf: str = "max_split_size_mb:256,roundup_power2_divisions:32",
) -> str:
    canonical = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    legacy = os.environ.get("PYTORCH_ALLOC_CONF", "").strip()

    alloc_conf = canonical or legacy or default_conf
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    # Backward-compatible alias for older launch scripts and custom user envs.
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    return alloc_conf


def apply_global_cuda_speed_knobs(
    torch_module: Any,
    allocator_default: str = "max_split_size_mb:256,roundup_power2_divisions:32",
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "cuda_available": False,
        "tf32": False,
        "matmul_precision": None,
        "cudnn_benchmark": False,
        "sdp": False,
        "allocator_conf": canonicalize_cuda_allocator_conf(allocator_default),
        "allocator_applied": False,
        "inductor_flags": False,
    }

    if not getattr(torch_module, "cuda", None) or not torch_module.cuda.is_available():
        return status

    status["cuda_available"] = True
    try:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True
        torch_module.backends.cudnn.benchmark = True
        torch_module.backends.cudnn.deterministic = False
        status["tf32"] = True
        status["cudnn_benchmark"] = True
    except Exception:
        pass

    try:
        if hasattr(torch_module, "set_float32_matmul_precision"):
            torch_module.set_float32_matmul_precision("high")
            status["matmul_precision"] = "high"
    except Exception:
        pass

    try:
        if hasattr(torch_module.nn.functional, "scaled_dot_product_attention"):
            torch_module.backends.cuda.enable_flash_sdp(True)
            torch_module.backends.cuda.enable_mem_efficient_sdp(True)
            torch_module.backends.cuda.enable_math_sdp(True)
            status["sdp"] = True
    except Exception:
        pass

    alloc_conf = status["allocator_conf"]
    try:
        if hasattr(torch_module, "_C") and hasattr(torch_module._C, "_accelerator_setAllocatorSettings"):
            torch_module._C._accelerator_setAllocatorSettings(alloc_conf)
            status["allocator_applied"] = True
        elif hasattr(torch_module.cuda, "memory") and hasattr(torch_module.cuda.memory, "_set_allocator_settings"):
            torch_module.cuda.memory._set_allocator_settings(alloc_conf)
            status["allocator_applied"] = True
    except Exception:
        pass

    # Torch Inductor config flags for maximum inference speed
    # Recommended by flux-fast repo and diffusers optimization docs.
    # These significantly speed up compiled transformer/VAE models.
    try:
        config = getattr(torch_module, "_inductor", None)
        if config is not None:
            if hasattr(config, "conv_1x1_as_mm"):
                config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
            if hasattr(config, "coordinate_descent_tuning"):
                config.coordinate_descent_tuning = True  # better autotuning algorithm
            if hasattr(config, "coordinate_descent_check_all_directions"):
                config.coordinate_descent_check_all_directions = True
            if hasattr(config, "epilogue_fusion"):
                config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls
            if hasattr(config, "force_fuse_int_mm_with_mul"):
                config.force_fuse_int_mm_with_mul = True  # fuse int matmul with mul
            if hasattr(config, "use_mixed_mm"):
                config.use_mixed_mm = True  # use mixed matmul for quantized models
            status["inductor_flags"] = True
    except Exception:
        pass

    # Dynamo config for quantized models (recommended by diffusers docs)
    # capture_dynamic_output_shape_ops handles dynamic outputs in compiled
    # quantized models; cache_size_limit avoids excessive recompilation.
    try:
        dynamo_config = getattr(torch_module, "_dynamo", None)
        if dynamo_config is not None:
            if hasattr(dynamo_config, "config"):
                dynamo_config.config.capture_dynamic_output_shape_ops = True
                if hasattr(dynamo_config.config, "cache_size_limit"):
                    dynamo_config.config.cache_size_limit = 1000
    except Exception:
        pass

    return status


def resolve_optimization_profile(
    requested_profile: Optional[str],
    device: str,
    cuda_runtime: Optional[str] = None,
) -> str:
    value = str(requested_profile or "").strip().lower()
    if value in OPTIMIZATION_PROFILES:
        return value
    if str(device).lower() == "cuda" and is_cuda13_runtime(cuda_runtime):
        return "max_speed"
    return "balanced"


def default_enable_windows_compile_probe(device: str, cuda_runtime: Optional[str] = None) -> bool:
    return os.name == "nt" and str(device).lower() == "cuda" and is_cuda13_runtime(cuda_runtime)


def get_torch_compile_probe_status(cache_key: str) -> str:
    return _TORCH_COMPILE_PROBE_STATUS.get(str(cache_key or ""), "untried")


def record_torch_compile_probe_result(cache_key: str, success: bool) -> None:
    key = str(cache_key or "")
    if not key:
        return
    _TORCH_COMPILE_PROBE_STATUS[key] = "success" if success else "failed"


def should_probe_torch_compile(
    cache_key: str,
    device: str,
    model_key: Optional[str],
    pipe: Optional[Any] = None,
    optimization_profile: str = "balanced",
    enable_windows_compile_probe: bool = False,
    cuda_runtime: Optional[str] = None,
) -> bool:
    if str(device).lower() != "cuda":
        return False
    if optimization_profile == "stability":
        return False
    if is_sdnq_model(model_key, pipe):
        return False

    status = get_torch_compile_probe_status(cache_key)
    if status == "failed":
        return False

    if os.name == "nt":
        if not enable_windows_compile_probe:
            # Allow compile on Windows if triton-windows is installed
            try:
                import triton  # noqa: F401
                pass  # Triton available, continue
            except ImportError:
                return False
        elif not is_cuda13_runtime(cuda_runtime):
            # No CUDA 13 and no triton-windows — check if triton is available anyway
            try:
                import triton  # noqa: F401
            except ImportError:
                return False

    return True


def should_use_attention_slicing(
    device: str,
    model_key: Optional[str],
    pipe: Optional[Any] = None,
    vram_gb: Optional[float] = None,
    optimization_profile: str = "balanced",
    oom_retry: bool = False,
) -> bool:
    if oom_retry:
        return True
    if str(device).lower() != "cuda":
        return True
    if optimization_profile == "stability":
        return True
    if optimization_profile == "max_speed":
        return False
    if is_sdnq_or_quantized(model_key, pipe):
        return False
    if vram_gb is not None and vram_gb <= 8.2:
        return True
    return False


def should_use_vae_slicing(
    device: str,
    vram_gb: Optional[float] = None,
    optimization_profile: str = "balanced",
    oom_retry: bool = False,
) -> bool:
    if oom_retry:
        return True
    if str(device).lower() != "cuda":
        return True
    if optimization_profile == "stability":
        return True
    _ = vram_gb
    return False


def should_use_vae_tiling(
    device: str,
    model_key: Optional[str],
    pipe: Optional[Any] = None,
    vram_gb: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    optimization_profile: str = "balanced",
) -> bool:
    if str(device).lower() != "cuda":
        return True
    if is_sdnq_or_quantized(model_key, pipe):
        return False
    if optimization_profile == "stability":
        return True

    long_edge = max(int(width or 0), int(height or 0))
    if long_edge >= 1536:
        return True
    _ = vram_gb
    return False


def resolve_optional_accelerators_enabled(
    requested_value: Optional[bool],
    optimization_profile: str,
) -> bool:
    if requested_value is not None:
        return bool(requested_value)
    return _read_bool_env(
        "UFIG_ENABLE_OPTIONAL_ACCELERATORS",
        default=(optimization_profile == "max_speed"),
    )


def describe_acceleration_stack(stack: Dict[str, Any]) -> str:
    ordered_keys = [
        "profile",
        "cuda_runtime",
        "tf32",
        "matmul_precision",
        "sdp",
        "allocator_conf",
        "autocast",
        "attention_slicing",
        "vae_slicing",
        "vae_tiling",
        "cpu_offload",
        "compile_probe",
        "compile_status",
        "cuda_graphs",
        "optional_accelerators",
    ]
    parts = []
    for key in ordered_keys:
        if key in stack:
            parts.append(f"{key}={stack[key]}")
    for key, value in stack.items():
        if key not in ordered_keys:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def choose_image_generation_mode(
    current_model: Optional[str],
    has_input_images: bool,
    enable_pose_preservation: bool = False,
) -> str:
    if is_flux_model(current_model):
        if has_input_images and not enable_pose_preservation:
            return "flux-img2img"
        return "txt2img"
    if is_zimage_model(current_model):
        if has_input_images:
            return "zimage-img2img"
        return "txt2img"
    return "txt2img"


def resolve_model_choice_for_device(
    model_choice: str,
    device: str,
    vram_gb: Optional[float] = None,
    gpu_name: Optional[str] = None,
    platform_system: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    choice = str(model_choice or "")
    if choice.strip():
        return choice, None

    resolved = resolve_default_flux_model_choice(
        device=device,
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        platform_system=platform_system,
    )
    return resolved, f"Resolved default FLUX path to {resolved} for the current Windows RTX 3070 profile."


def resolve_zimage_img2img_steps(steps: int, minimum_steps: int = 8) -> Tuple[int, bool]:
    safe_steps = int(steps)
    if safe_steps < minimum_steps:
        return minimum_steps, True
    return safe_steps, False


def select_requirements_file(base_dir: str, is_windows: bool, cuda_available: bool) -> str:
    lockfile = "requirements-lock-cu130.txt"
    if is_windows and os.path.exists(os.path.join(base_dir, lockfile)):
        return lockfile
    return "requirements.txt"


def is_cuda13_runtime(cuda_runtime: Optional[str]) -> bool:
    if not cuda_runtime:
        return False
    return str(cuda_runtime).startswith("13.")


def compute_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dependency_profile_metadata(requirements_file: str, requirements_hash: str) -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "requirements_file": requirements_file,
        "requirements_hash": requirements_hash,
    }


def is_dependency_metadata_current(existing: Optional[dict], expected: dict) -> bool:
    if not isinstance(existing, dict):
        return False
    for key, value in expected.items():
        if existing.get(key) != value:
            return False
    return True
