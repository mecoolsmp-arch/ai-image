from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from comfyui_app.model_resolver import ModelResolverError

logger = logging.getLogger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


def _save_frame(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is None:
        raise ModelResolverError("OpenCV is not available, so frames cannot be written.")
    cv2.imwrite(str(output_path), image)


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_n: int = 1,
    max_frames: int | None = None,
) -> list[str]:
    source = Path(video_path)
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if cv2 is not None:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ModelResolverError(f"Could not open the video file: {source}")
        saved: list[str] = []
        frame_index = 0
        output_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % max(1, every_n) == 0:
                    output_path = target_dir / f"frame_{output_index:06d}.png"
                    _save_frame(frame, output_path)
                    saved.append(str(output_path))
                    output_index += 1
                    if max_frames is not None and len(saved) >= max_frames:
                        break
                frame_index += 1
        finally:
            capture.release()
        return saved

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ModelResolverError(
            "OpenCV is not installed and ffmpeg was not found on PATH, so video frames cannot be extracted."
        )

    pattern = target_dir / "frame_%06d.png"
    select_filter = f"select='not(mod(n\\,{max(1, every_n)}))'"
    command = [
        ffmpeg,
        "-i",
        str(source),
        "-vf",
        select_filter,
        "-vsync",
        "vfr",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(pattern))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ModelResolverError("ffmpeg could not extract frames from the uploaded video.")
    saved = sorted(str(path) for path in target_dir.glob("frame_*.png"))
    if max_frames is not None:
        saved = saved[:max_frames]
    return saved
