"""Compatibility helpers for third-party JSON serialization edge cases."""

from __future__ import annotations

import contextlib
import json
from typing import Generator

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in normal app installs
    _np = None

try:
    import torch as _torch
except Exception:  # pragma: no cover - torch is present in normal app installs
    _torch = None

_original_json_encoder_default = json.JSONEncoder.default
_patched = False


def _is_dtype_like(obj) -> bool:
    if _torch is not None and isinstance(obj, _torch.dtype):
        return True
    if _np is not None and isinstance(obj, _np.dtype):
        return True
    # Triton language dtypes (used in autotuner configs)
    try:
        import triton.language as _tl
        if isinstance(obj, _tl.dtype):
            return True
    except Exception:
        pass
    return False


def _json_encoder_default_patch(self, obj):
    if _is_dtype_like(obj):
        return str(obj)
    return _original_json_encoder_default(self, obj)


def patch_json_dtype_serialization() -> None:
    """Let json.dumps serialize torch/numpy/triton dtype objects as strings.

    Triton's autotuner writes benchmark metadata with the standard library
    ``json.dumps`` and may include Triton dtype instances in config dicts.
    Patching the encoder class keeps the behavior process-wide without
    mutating CPython's cached encoder instance.
    """
    global _patched
    if _patched:
        return
    json.JSONEncoder.default = _json_encoder_default_patch
    _patched = True


def unpatch_json_dtype_serialization() -> None:
    """Restore the original json.JSONEncoder.default implementation."""
    global _patched
    if not _patched:
        return
    json.JSONEncoder.default = _original_json_encoder_default
    _patched = False


@contextlib.contextmanager
def json_dtype_patch_context() -> Generator[None, None, None]:
    """Temporarily enable dtype-aware JSON serialization inside a block."""
    patch_json_dtype_serialization()
    try:
        yield
    finally:
        unpatch_json_dtype_serialization()
