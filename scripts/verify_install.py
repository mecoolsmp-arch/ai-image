import os
import subprocess
import sys
import argparse
import platform

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


CRITICAL_RESOLVER_ARGS = [
    "transformers==4.57.3",
    "huggingface_hub[hf_xet]==0.36.2",
    "hf_xet",
]


def run_pip_check() -> bool:
    print("Running pip dependency check...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[OK] pip check")
        return True
    print("[FAIL] pip check")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return False


def check_imports() -> bool:
    print("Checking critical imports...")
    checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("gradio", "Gradio"),
        ("PIL", "Pillow"),
        ("accelerate", "Accelerate"),
        ("hf_xet", "HF Xet"),
    ]

    ok = True
    for module_name, label in checks:
        try:
            __import__(module_name)
            print(f"[OK] {label}")
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")
            ok = False
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401
        print("[OK] Transformers AutoModel/AutoProcessor")
    except Exception as exc:
        print(f"[FAIL] Transformers AutoModel/AutoProcessor: {exc}")
        ok = False
    return ok


def print_runtime_matrix() -> bool:
    print("Runtime matrix:")
    matrix_ok = True

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_runtime = torch.version.cuda
        print(f"  torch={torch.__version__}")
        print(f"  cuda_available={cuda_available}")
        print(f"  torch_cuda_runtime={cuda_runtime}")
        print(f"  optimization_profile={os.getenv('UFIG_OPTIMIZATION_PROFILE', 'default')}")
        allocator_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF") or os.getenv("PYTORCH_ALLOC_CONF") or "unset"
        print(f"  cuda_allocator_conf={allocator_conf}")
        print(f"  hf_symlink_warning_disabled={os.getenv('HF_HUB_DISABLE_SYMLINKS_WARNING', 'unset')}")
        print(f"  sdnq_torch_compile={os.getenv('SDNQ_USE_TORCH_COMPILE', 'auto')}")
        print(f"  sdnq_triton_mm={os.getenv('SDNQ_USE_TRITON_MM', 'auto')}")
        print(f"  sdnq_compile_kwargs={os.getenv('SDNQ_COMPILE_KWARGS', 'default')}")
        print(f"  optional_accelerators={os.getenv('UFIG_ENABLE_OPTIONAL_ACCELERATORS', 'default')}")
        print(f"  torch_inductor_cache={os.getenv('TORCHINDUCTOR_CACHE_DIR', 'unset')}")
        if cuda_available:
            print(f"  gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  [FAIL] torch runtime: {exc}")
        return False

    # Check for core speedup packages (now required installs)
    try:
        import triton  # noqa: F401
        print("  triton=available (torch.compile + SDNQ optimizations enabled)")
    except ImportError:
        print("  triton=MISSING (install triton-windows>=3.6,<3.7 for torch.compile + SDNQ speedups)")

    try:
        from sageattention import sageattn  # noqa: F401
        print("  sageattention=available (2-5x attention speedup)")
    except ImportError:
        print("  sageattention=MISSING (install sageattention for attention speedup)")

    optional_accel_enabled = os.getenv("UFIG_ENABLE_OPTIONAL_ACCELERATORS", "0") == "1"
    if optional_accel_enabled:
        try:
            import xformers  # noqa: F401

            print("  xformers=available")
        except Exception as exc:
            print(f"  xformers=unavailable ({exc})")
    else:
        print("  xformers=optional (disabled)")

    return matrix_ok


def run_resolver_checks() -> bool:
    print("Running dependency resolver dry-run checks...")
    ok = True

    lock_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--no-deps",
        "-r",
        "requirements-lock-cu130.txt",
    ]
    lock_result = subprocess.run(lock_cmd, capture_output=True, text=True)
    if lock_result.returncode == 0:
        print("[OK] lockfile dry-run --no-deps")
    else:
        print("[FAIL] lockfile dry-run --no-deps")
        if lock_result.stdout.strip():
            print(lock_result.stdout.strip())
        if lock_result.stderr.strip():
            print(lock_result.stderr.strip())
        ok = False

    trio_cmd = [sys.executable, "-m", "pip", "install", "--dry-run", *CRITICAL_RESOLVER_ARGS]
    trio_result = subprocess.run(trio_cmd, capture_output=True, text=True)
    if trio_result.returncode == 0:
        print("[OK] critical trio resolver dry-run")
    else:
        print("[FAIL] critical trio resolver dry-run")
        if trio_result.stdout.strip():
            print(trio_result.stdout.strip())
        if trio_result.stderr.strip():
            print(trio_result.stderr.strip())
        ok = False

    return ok


def parse_args():
    parser = argparse.ArgumentParser(description="Verify runtime/install health.")
    parser.add_argument(
        "--strict-resolver",
        action="store_true",
        help="Run pip resolver dry-run checks for lockfile and critical package trio.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ok = True
    if not run_pip_check():
        ok = False
    if not check_imports():
        ok = False
    if not print_runtime_matrix():
        ok = False
    if args.strict_resolver and not run_resolver_checks():
        ok = False

    if ok:
        print("\nSUCCESS: verification passed")
        return 0

    print("\nFAIL: verification found issues")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
