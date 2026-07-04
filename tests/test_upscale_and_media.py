from __future__ import annotations

from pathlib import Path

from comfyui_app import app, batch, video_frames
from comfyui_app.generation import GenerationResult
from comfyui_app.workflow_builder import build_esrgan_upscale_prompt, build_rtx_upscale_prompt


def test_build_rtx_upscale_prompt_serializes_flat_inputs() -> None:
    prompt = build_rtx_upscale_prompt(image="input.png", resize_type="scale by multiplier", scale=2.0, quality="ULTRA")

    assert prompt["2"]["class_type"] == "RTXVideoSuperResolution"
    assert prompt["2"]["inputs"] == {
        "images": ["1", 0],
        "resize_type": "scale by multiplier",
        "scale": 2.0,
        "quality": "ULTRA",
    }


def test_build_esrgan_upscale_prompt_uses_core_upscale_nodes() -> None:
    prompt = build_esrgan_upscale_prompt(image="input.png", target_width=2048, target_height=2048, resize_type="target dimensions")

    class_types = {node["class_type"] for node in prompt.values()}
    assert {"LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScaleToTotalPixels", "SaveImage"} <= class_types
    scale_node_id = next(node_id for node_id, node in prompt.items() if node["class_type"] == "ImageScaleToTotalPixels")
    assert prompt[scale_node_id]["inputs"]["resolution_steps"] == 1


def test_process_folder_creates_timestamped_run_dir_and_honors_cancel(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.png").write_bytes(b"one")
    (source / "two.png").write_bytes(b"two")
    outputs: list[Path] = []

    def gen_fn(image_path: Path, prompt: str, negative: str, output_dir: Path) -> GenerationResult:
        result_path = output_dir / f"{image_path.stem}_out.png"
        result_path.write_bytes(b"out")
        outputs.append(result_path)
        if len(outputs) == 1:
            batch.request_cancel()
        return GenerationResult(image_path=result_path, status="ok")

    summary = batch.process_folder(source, tmp_path / "results", "", "", gen_fn)
    run_dir = Path(str(summary["output_dir"]))

    assert run_dir.name.startswith("batch_")
    assert run_dir.exists()
    assert summary["count"] == 1
    assert len(summary["results"]) == 1
    assert outputs[0].parent == run_dir
    batch.clear_cancel()


def test_process_folder_clears_stale_cancel_flag_before_start(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.png").write_bytes(b"one")
    batch.request_cancel()

    def gen_fn(image_path: Path, prompt: str, negative: str, output_dir: Path) -> GenerationResult:
        result_path = output_dir / f"{image_path.stem}_out.png"
        result_path.write_bytes(b"out")
        return GenerationResult(image_path=result_path, status="ok")

    summary = batch.process_folder(source, tmp_path / "results", "", "", gen_fn)

    assert summary["count"] == 1
    assert len(summary["results"]) == 1
    batch.clear_cancel()


def test_iter_process_folder_stops_when_cancelled_mid_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.png").write_bytes(b"one")
    (source / "two.png").write_bytes(b"two")
    seen: list[str] = []

    def gen_fn(image_path: Path, prompt: str, negative: str, output_dir: Path) -> GenerationResult:
        result_path = output_dir / f"{image_path.stem}_out.png"
        result_path.write_bytes(b"out")
        seen.append(image_path.name)
        batch.request_cancel()
        return GenerationResult(image_path=result_path, status="ok")

    updates = list(batch.iter_process_folder(source, tmp_path / "results", "", "", gen_fn))

    assert len(updates) >= 1
    assert updates[-1]["count"] == 1
    assert seen == ["one.png"]
    batch.clear_cancel()


def test_build_frames_to_video_command_includes_audio_when_present(monkeypatch) -> None:
    class _FakeFFmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    monkeypatch.setattr(video_frames, "imageio_ffmpeg", _FakeFFmpeg())
    command = video_frames.build_frames_to_video_command("frame_%06d.png", "out.mp4", 24.0, audio_source="source.mov", has_audio=True)

    assert command[:6] == ["ffmpeg", "-y", "-framerate", "24", "-i", "frame_%06d.png"]
    assert "-map" in command and "0:v" in command and "1:a?" in command
    assert command[-1] == "out.mp4"


def test_build_frames_to_video_command_omits_audio_when_missing(monkeypatch) -> None:
    class _FakeFFmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    monkeypatch.setattr(video_frames, "imageio_ffmpeg", _FakeFFmpeg())
    command = video_frames.build_frames_to_video_command("frame_%06d.png", "out.mp4", 24.0, audio_source=None, has_audio=False)

    assert "-map" not in command
    assert "aac" not in command
    assert command[-1] == "out.mp4"


def test_image_edit_handler_routes_to_depth_path_only_when_enabled(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_run_edit(*args, **kwargs):
        calls.append(("edit", args, kwargs))
        return GenerationResult(image_path=Path("edit.png"), status="edit")

    def fake_run_depth_edit(*args, **kwargs):
        calls.append(("depth", args, kwargs))
        return GenerationResult(image_path=Path("depth.png"), status="depth")

    monkeypatch.setattr(app, "run_edit", fake_run_edit)
    monkeypatch.setattr(app, "run_depth_edit", fake_run_depth_edit)

    result = app._edit_handler("input.png", None, "prompt", "negative", "out", 4, 1.0, 1.0, 0, "default", False, False, False)
    assert result == ("edit.png", "edit")
    assert calls[0][0] == "edit"

    calls.clear()
    result = app._edit_handler("input.png", "reference.png", "prompt", "negative", "out", 4, 1.0, 1.0, 0, "default", False, False, True)
    assert result == ("depth.png", "depth")
    assert calls[0][0] == "depth"
