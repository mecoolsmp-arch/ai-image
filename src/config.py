"""Application-wide environment configuration and CUDA setup.

Centralizes environment variable defaults, cache paths, and CUDA 13
optimization knobs that were previously scattered across app.py.
"""

import logging
import os
import sys
import warnings

# Suppress noisy xformers warnings when the installed wheel is incompatible
# with the current PyTorch / Python build.  SageAttention / SDPA are used
# as the primary attention backends, so a broken xformers install is harmless.
logging.getLogger("xformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Importing from timm\.models\..* is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Overwriting tiny_vit_.* in registry.*", category=UserWarning)

# Windows-specific fix for asyncio ConnectionResetError (WinError 10054)
if sys.platform == "win32":
    import asyncio

    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except ImportError:
        pass
    else:
        def _silence_connection_lost(original_func):
            def wrapper(self, exc=None):
                if isinstance(exc, ConnectionResetError):
                    exc = None
                try:
                    return original_func(self, exc)
                except ConnectionResetError:
                    pass
            return wrapper

        _ProactorBasePipeTransport._call_connection_lost = _silence_connection_lost(
            _ProactorBasePipeTransport._call_connection_lost
        )


def _configure_sdnq_environment():
    """Configure SDNQ/Triton environment variables for CUDA 13."""
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401
        except ImportError:
            os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")
            os.environ.setdefault("SDNQ_USE_TRITON_MM", "0")

    # Force SDNQ to use Triton-based INT8 matmul when available.
    # The default torch._int_mm goes through oneDNN internally, which lacks
    # CUDA dispatch kernels for onednn.qlinear_prepack.
    try:
        import triton  # noqa: F401
        os.environ.setdefault("SDNQ_USE_TRITON_MM", "1")
    except ImportError:
        pass

    # On Windows, override SDNQ's torch.compile to use the 'eager' backend
    # to avoid inductor hangs during Triton kernel compilation.
    if sys.platform == "win32":
        os.environ.setdefault("SDNQ_COMPILE_KWARGS", '{"backend": "eager"}')


_configure_sdnq_environment()

import torch  # noqa: E402 — must import after SDNQ env is configured

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPENDENCY_CHECK_FLAG = os.path.join(BASE_DIR, ".dependencies_verified")
SKIP_DEPENDENCY_CHECK = os.environ.get("SKIP_DEPENDENCY_CHECK", "0") == "1"

STATE_DIR = os.path.join(BASE_DIR, "user_state")
os.makedirs(STATE_DIR, exist_ok=True)

LORA_DIR = os.path.join(BASE_DIR, "loras")
os.makedirs(LORA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Cache configuration (override via environment variables)
# ---------------------------------------------------------------------------
CACHE_ROOT = os.environ.get("UFIG_CACHE_DIR", os.path.join(BASE_DIR, "cache"))
HF_HOME = os.environ.get("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))

# ---------------------------------------------------------------------------
# API Keys (override via environment variables)
# ---------------------------------------------------------------------------
CIVITAI_API_TOKEN = os.environ.get("CIVITAI_API_TOKEN", "2e9ca58a0eac002a139f12ec096d4bd8")

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(HF_HOME, "xet"))
os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(HF_HOME, "assets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_ROOT, "torch"))
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(CACHE_ROOT, "gradio"))


# ---------------------------------------------------------------------------
# CUDA 13 / RTX 3070 optimizations
# ---------------------------------------------------------------------------
from src.runtime_policies import (  # noqa: E402
    apply_global_cuda_speed_knobs,
    canonicalize_cuda_allocator_conf,
    default_enable_windows_compile_probe,
    is_cuda13_runtime,
    resolve_optimization_profile,
    resolve_optional_accelerators_enabled,
)

RUNTIME_CUDA13 = is_cuda13_runtime(getattr(torch.version, "cuda", None))

DEFAULT_OPTIMIZATION_PROFILE = resolve_optimization_profile(
    requested_profile=os.environ.get("UFIG_OPTIMIZATION_PROFILE"),
    device="cuda" if torch.cuda.is_available() else "cpu",
    cuda_runtime=getattr(torch.version, "cuda", None),
)

DEFAULT_WINDOWS_COMPILE_PROBE = default_enable_windows_compile_probe(
    device="cuda" if torch.cuda.is_available() else "cpu",
    cuda_runtime=getattr(torch.version, "cuda", None),
)

DEFAULT_OPTIONAL_ACCELERATORS = resolve_optional_accelerators_enabled(
    requested_value=None,
    optimization_profile=DEFAULT_OPTIMIZATION_PROFILE,
)

CUDA_SPEED_STATUS = apply_global_cuda_speed_knobs(torch)
canonicalize_cuda_allocator_conf()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except Exception:
        return default


def read_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
