from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import subprocess
from typing import Final

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelTier:
    diffusion: str
    text_encoder: str
    use_tiled_decode: bool
    extra_launch_flags: list[str]
    label: str


_GIB: Final[float] = 1024.0**3


def _parse_nvidia_smi_output(output: str) -> tuple[float, str] | None:
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            continue
        memory_text, device_name = parts[0], parts[1]
        match = re.search(r"([\d,.]+)\s*([A-Za-z]+)?", memory_text)
        if not match:
            continue
        number = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "MiB").lower()
        if unit in {"mib", "mb"}:
            total_gb = number / 1024.0
        else:
            total_gb = number
        return total_gb, device_name
    return None


def detect_vram() -> tuple[float, str, bool]:
    if torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        properties = torch.cuda.get_device_properties(0)  # type: ignore[union-attr]
        total_gb = float(properties.total_memory) / _GIB
        return total_gb, str(properties.name), True

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - platform fallback
        logger.debug("nvidia-smi probe failed: %s", exc)
        return 0.0, "No NVIDIA GPU detected", False

    if result.returncode != 0 or not result.stdout.strip():
        return 0.0, "No NVIDIA GPU detected", False

    parsed = _parse_nvidia_smi_output(result.stdout)
    if parsed is None:
        return 0.0, "No NVIDIA GPU detected", False

    total_gb, device_name = parsed
    return total_gb, device_name, False


def select_tier(vram_gb: float) -> ModelTier:
    if vram_gb >= 16.0:
        return ModelTier(
            diffusion="flux2_fp8",
            text_encoder="flux2_full",
            use_tiled_decode=False,
            extra_launch_flags=[],
            label="16 GB or more",
        )
    if vram_gb >= 10.0:
        return ModelTier(
            diffusion="flux2_fp8",
            text_encoder="flux2_fp4",
            use_tiled_decode=False,
            extra_launch_flags=[],
            label="10 to 15 GB",
        )
    if vram_gb >= 7.0:
        return ModelTier(
            diffusion="flux2_fp8",
            text_encoder="flux2_fp4",
            use_tiled_decode=True,
            extra_launch_flags=[],
            label="7 to 9 GB",
        )
    if vram_gb >= 5.0:
        return ModelTier(
            diffusion="flux2_gguf_q4_k_m",
            text_encoder="flux2_fp4",
            use_tiled_decode=True,
            extra_launch_flags=["--lowvram"],
            label="5 to 6 GB",
        )
    if vram_gb >= 4.0:
        return ModelTier(
            diffusion="flux2_gguf_q3_k_m",
            text_encoder="flux2_fp4",
            use_tiled_decode=True,
            extra_launch_flags=["--lowvram"],
            label="4 to 4.9 GB",
        )
    return ModelTier(
        diffusion="flux2_gguf_q2_k",
        text_encoder="flux2_fp4",
        use_tiled_decode=True,
        extra_launch_flags=["--lowvram"],
        label="under 4 GB",
    )
