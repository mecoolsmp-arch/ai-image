from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from comfyui_app.model_resolver import ModelResolverError
from comfyui_app.generation import GenerationResult

logger = logging.getLogger(__name__)


class SingleImageEditFn(Protocol):
    def __call__(
        self,
        input_image_path: Path,
        prompt: str,
        negative: str,
        output_dir: Path,
    ) -> GenerationResult | Path:
        ...


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    prompt: str,
    negative: str,
    gen_fn: SingleImageEditFn,
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
) -> dict[str, object]:
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise ModelResolverError(f"The input folder does not exist: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    failures: list[str] = []
    for file_path in sorted(source_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in exts:
            continue
        try:
            result = gen_fn(file_path, prompt, negative, target_dir)
            output_path = result.image_path if isinstance(result, GenerationResult) else Path(result)
            processed.append(str(output_path))
        except Exception as exc:
            failures.append(f"{file_path.name}: {exc}")
            logger.exception("Failed to process %s", file_path)
    message = "No images found in that folder." if not processed and not failures else f"Processed {len(processed)} files."
    return {
        "count": len(processed),
        "failures": failures,
        "results": processed,
        "message": message,
    }
