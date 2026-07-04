"""
Process-wide Face Swap configuration.

Threading two extra arguments (auto-swap enable, similarity threshold)
through every Single / Batch / Video UI entry point would require
touching 30+ parameter signatures in ``app.py``. The clean alternative
is a small singleton that the UI writes when the user toggles the
controls in the Face Swap tab, and every post-generation hook reads.

The config is intentionally tiny and threadsafe — Gradio request
threads can all read the current value without locking because a bool
+ float read is atomic on CPython.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class FaceSwapConfig:
    """Runtime-configurable Face Swap post-processing settings."""

    # When True, every generated image is run
    # through :func:`post_process_with_library` before being displayed.
    auto_swap_from_library: bool = False

    # Cosine-similarity cutoff on the normalized InsightFace embedding.
    # See :meth:`FaceSwapHelper.auto_swap_with_library`.
    similarity_threshold: float = 0.35


_CONFIG = FaceSwapConfig()
_CONFIG_LOCK = threading.Lock()


def get_config() -> FaceSwapConfig:
    """Return a copy of the current configuration (safe to read async)."""
    with _CONFIG_LOCK:
        return FaceSwapConfig(
            auto_swap_from_library=_CONFIG.auto_swap_from_library,
            similarity_threshold=_CONFIG.similarity_threshold,
        )


def set_config(
    *,
    auto_swap_from_library: bool | None = None,
    similarity_threshold: float | None = None,
) -> FaceSwapConfig:
    """Update one or more fields, returning the new config snapshot."""
    with _CONFIG_LOCK:
        if auto_swap_from_library is not None:
            _CONFIG.auto_swap_from_library = bool(auto_swap_from_library)
        if similarity_threshold is not None:
            _CONFIG.similarity_threshold = float(similarity_threshold)
        return FaceSwapConfig(
            auto_swap_from_library=_CONFIG.auto_swap_from_library,
            similarity_threshold=_CONFIG.similarity_threshold,
        )


def post_process_with_library(image, device: str = "cuda"):
    """
    If auto-swap is on, run the image through the library-matching
    inswapper pass and return the possibly-modified image. Otherwise
    return the input unchanged.

    Any failure is logged and swallowed so an unexpected library issue
    never crashes the whole generation pipeline — the original image
    will be delivered to the user instead.
    """
    if image is None:
        return image
    cfg = get_config()
    if not cfg.auto_swap_from_library:
        return image

    try:
        from src.image.faceswap_helper import get_faceswap_helper

        swapper = get_faceswap_helper(device=device)
        result, matched = swapper.auto_swap_with_library(
            image,
            similarity_threshold=cfg.similarity_threshold,
        )
        if matched:
            print(
                f"  [face-swap] Auto-swapped characters from library: "
                f"{', '.join(matched)}"
            )
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"  [face-swap] Auto library swap failed: {exc}")
        return image
