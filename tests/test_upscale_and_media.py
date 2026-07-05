from __future__ import annotations

import io
import json
import struct
from pathlib import Path

from comfyui_app import app, batch, video_frames
from comfyui_app import comfy_client
from comfyui_app.generation import GenerationResult
from comfyui_app.workflow_builder import build_esrgan_upscale_prompt, build_rtx_upscale_prompt


def _assert_restored_value(actual: object, wanted: object, *, path_like: bool = False) -> None:
    if isinstance(wanted, dict) and wanted.get("__type__") == "update":
        assert isinstance(actual, dict)
        assert actual.get("__type__") == "update"
        wanted_value = wanted.get("value")
        if path_like and isinstance(wanted_value, str):
            actual_value = actual.get("value")
            assert isinstance(actual_value, str)
            assert Path(actual_value).name == Path(wanted_value).name
        else:
            assert actual.get("value") == wanted_value
        if "visible" in wanted:
            assert actual.get("visible") == wanted["visible"]
        return
    assert actual == wanted


def test_build_rtx_upscale_prompt_serializes_flat_inputs() -> None:
    prompt = build_rtx_upscale_prompt(image="input.png", resize_type="scale by multiplier", scale=2.0, quality="ULTRA")

    assert prompt["2"]["class_type"] == "RTXVideoSuperResolution"
    assert prompt["2"]["inputs"] == {
        "images": ["1", 0],
        "resize_type": "scale by multiplier",
        "resize_type.scale": 2.0,
        "quality": "ULTRA",
    }


def test_build_rtx_upscale_prompt_nests_target_dimensions() -> None:
    prompt = build_rtx_upscale_prompt(image="input.png", resize_type="target dimensions", width=1920, height=1080)

    assert prompt["2"]["inputs"]["resize_type"] == "target dimensions"
    assert prompt["2"]["inputs"]["resize_type.width"] == 1920
    assert prompt["2"]["inputs"]["resize_type.height"] == 1080
    assert "resize_type.scale" not in prompt["2"]["inputs"]
    assert "scale" not in prompt["2"]["inputs"]


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


def test_comfy_client_surfaces_live_preview_frames(monkeypatch) -> None:
    from PIL import Image

    preview = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(preview, format="PNG")
    payload = struct.pack(">II", 1, 2) + preview.getvalue()

    class FakeWS:
        def __init__(self) -> None:
            self.messages = [
                payload,
                json.dumps({"type": "executing", "data": {"prompt_id": "p1", "node": None}}),
            ]

        def recv(self):
            return self.messages.pop(0)

        def close(self):
            return None

    monkeypatch.setattr(comfy_client, "create_connection", lambda *args, **kwargs: FakeWS())
    client = comfy_client.ComfyClient("127.0.0.1", 8188)
    previews: list[object] = []

    client.wait_for_completion("p1", preview_callback=previews.append)

    assert len(previews) == 1
    assert previews[0].size == (4, 4)


def test_image_edit_handler_routes_to_depth_path_only_when_enabled(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_run_edit(*args, **kwargs):
        calls.append(("edit", args, kwargs))
        return GenerationResult(image_path=Path("edit.png"), status="edit")

    def fake_run_depth_edit(*args, **kwargs):
        calls.append(("depth", args, kwargs))
        return GenerationResult(image_path=Path("depth.png"), status="depth", preview_path=Path("depth-map.png"))

    monkeypatch.setattr(app, "run_edit", fake_run_edit)
    monkeypatch.setattr(app, "run_depth_edit", fake_run_depth_edit)

    result = list(app._edit_handler("input.png", None, "prompt", "negative", "out", 4, 1.0, 1.0, 0, "int8", False, False, False, False, False))
    assert result == [("edit.png", None, None, "edit")]
    assert calls[0][0] == "edit"

    calls.clear()
    result = list(app._edit_handler("input.png", "reference.png", "prompt", "negative", "out", 4, 1.0, 1.0, 0, "int8", False, False, False, True, False))
    assert result == [("depth.png", None, "depth-map.png", "depth")]
    assert calls[0][0] == "depth"


def test_app_default_engine_is_int8() -> None:
    assert app.DEFAULT_ENGINE_VALUE == "int8"
    assert app.ENGINE_CHOICES[0] == ("INT8 (fastest on Ampere - default)", "int8")


def test_app_exposes_model_cleanup_controls() -> None:
    demo = app.build_app()
    button_values = [
        component.get("props", {}).get("value")
        for component in demo.config["components"]
        if isinstance(component, dict) and component.get("type") == "button"
    ]
    labels = [
        component.get("props", {}).get("label")
        for component in demo.config["components"]
        if isinstance(component, dict)
    ]
    assert "Remove unused / duplicate models" in button_values
    assert "Confirm removal" in button_values
    assert any("TeaCache speedup" in str(label or "") for label in labels)
    assert not any(label and "base instead" in str(label).lower() for label in labels)


def test_app_configures_browser_state_for_persistence() -> None:
    demo = app.build_app()
    browser_states = [component for component in demo.config["components"] if isinstance(component, dict) and component.get("type") == "browserstate"]

    assert len(browser_states) == 1
    props = browser_states[0]["props"]
    assert props["storage_key"] == app.UI_STATE_STORAGE_KEY
    assert props["secret"] == app.UI_STATE_SECRET
    assert props["default_value"]["shared_prompt"] == ""
    assert props["default_value"]["shared_negative"] == ""


def test_shared_prompt_wiring_uses_blur_and_tab_select() -> None:
    demo = app.build_app()
    components = [component for component in demo.config["components"] if isinstance(component, dict)]
    dependencies = demo.config["dependencies"]

    prompt_ids = [component["id"] for component in components if component.get("type") == "textbox" and component.get("props", {}).get("label") == "Prompt"]
    negative_ids = [component["id"] for component in components if component.get("type") == "textbox" and component.get("props", {}).get("label") == "Negative prompt"]
    tab_ids = [component["id"] for component in components if component.get("type") == "tabitem"][:4]

    assert len(prompt_ids) == 4
    assert len(negative_ids) == 4
    assert len(tab_ids) == 4

    for textbox_id in prompt_ids + negative_ids:
        assert any(dep["targets"] == [(textbox_id, "blur")] for dep in dependencies)
        assert not any(dep["targets"] == [(textbox_id, "input")] for dep in dependencies)

    expected_outputs = [
        [prompt_ids[0], negative_ids[0]],
        [prompt_ids[1], negative_ids[1]],
        [prompt_ids[2], negative_ids[2]],
        [prompt_ids[3], negative_ids[3]],
    ]
    for tab_id, outputs in zip(tab_ids, expected_outputs, strict=True):
        assert any(dep["targets"] == [(tab_id, "select")] and dep["outputs"] == outputs for dep in dependencies)


def test_restore_ui_state_shares_prompts_and_skips_missing_files(tmp_path: Path) -> None:
    def update(*, value=None, visible=None):
        data = {"__type__": "update"}
        if value is not None:
            data["value"] = value
        if visible is not None:
            data["visible"] = visible
        return data

    input_image = tmp_path / "input.png"
    input_image.write_bytes(b"image")
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"reference")
    video_input = tmp_path / "video.mp4"
    video_input.write_bytes(b"video")
    upscale_image = tmp_path / "upscale.png"
    upscale_image.write_bytes(b"upscale")
    video_upscale_input = tmp_path / "video-upscale.mp4"
    video_upscale_input.write_bytes(b"video-upscale")

    state = {
        "shared_prompt": "PROMPT",
        "shared_negative": "NEGATIVE",
        "edit_image": str(input_image),
        "edit_reference": str(reference_image),
        "edit_output": "EDIT_OUTPUT",
        "edit_steps": 101,
        "edit_cfg": 1.01,
        "edit_megapixels": 1.11,
        "edit_seed": 111,
        "edit_engine": "nunchaku_int4",
        "edit_live_preview": True,
        "edit_consistency": False,
        "edit_consistency_strength": 1.23,
        "edit_compile": True,
        "edit_mrflow": False,
        "edit_teacache": True,
        "edit_depth_lock": True,
        "video_input": str(video_input),
        "video_output": "VIDEO_OUTPUT",
        "video_steps": 202,
        "video_cfg": 2.02,
        "video_megapixels": 2.22,
        "video_seed": 222,
        "video_engine": "int8",
        "video_compile": False,
        "video_mrflow": True,
        "video_every_n": 3,
        "video_max_frames": 4,
        "video_frame_dir": "VIDEO_FRAMES",
        "batch_input": "BATCH_INPUT",
        "batch_output": "BATCH_OUTPUT",
        "batch_steps": 303,
        "batch_cfg": 3.03,
        "batch_megapixels": 3.33,
        "batch_seed": 333,
        "batch_engine": "nunchaku_int4",
        "batch_consistency": True,
        "batch_consistency_strength": 0.45,
        "batch_compile": False,
        "batch_teacache": True,
        "batch_mrflow": False,
        "t2i_output": "T2I_OUTPUT",
        "t2i_width": 512,
        "t2i_height": 768,
        "t2i_steps": 404,
        "t2i_cfg": 4.04,
        "t2i_seed": 444,
        "t2i_engine": "nunchaku_int4",
        "t2i_live_preview": False,
        "t2i_consistency": True,
        "t2i_consistency_strength": 1.66,
        "t2i_compile": True,
        "t2i_mrflow": False,
        "t2i_teacache": True,
        "upscale_image": str(upscale_image),
        "upscale_output": "UPSCALE_OUTPUT",
        "upscale_upscaler": "esrgan",
        "upscale_scale": 2.5,
        "upscale_quality": "LOW",
        "upscale_input_folder": "UPSCALE_INPUT",
        "upscale_folder_output": "UPSCALE_FOLDER_OUTPUT",
        "upscale_folder_upscaler": "esrgan",
        "upscale_folder_scale": 1.5,
        "upscale_folder_quality": "MEDIUM",
        "video_upscale_input": str(video_upscale_input),
        "video_upscale_output": "VIDEO_UPSCALE_OUTPUT",
        "video_upscale_upscaler": "esrgan",
        "video_upscale_scale": 1.75,
        "video_upscale_quality": "HIGH",
    }

    restored = app._restore_ui_state(state)

    expected = [
        "PROMPT",
        "PROMPT",
        "PROMPT",
        "PROMPT",
        "NEGATIVE",
        "NEGATIVE",
        "NEGATIVE",
        "NEGATIVE",
        update(value=str(input_image)),
        update(value=str(reference_image), visible=True),
        "EDIT_OUTPUT",
        101,
        1.01,
        1.11,
        111,
        "nunchaku_int4",
        True,
        update(visible=True),
        False,
        update(value=1.23, visible=False),
        True,
        False,
        True,
        True,
        update(visible=True),
        update(visible=True),
        update(value=str(video_input)),
        "VIDEO_OUTPUT",
        202,
        2.02,
        2.22,
        222,
        "int8",
        False,
        True,
        3,
        4,
        "VIDEO_FRAMES",
        "BATCH_INPUT",
        "BATCH_OUTPUT",
        303,
        3.03,
        3.33,
        333,
        "nunchaku_int4",
        True,
        update(value=0.45, visible=True),
        False,
        True,
        False,
        "T2I_OUTPUT",
        512,
        768,
        404,
        4.04,
        444,
        "nunchaku_int4",
        False,
        True,
        update(value=1.66, visible=True),
        True,
        False,
        True,
        update(value=str(upscale_image)),
        "UPSCALE_OUTPUT",
        "esrgan",
        2.5,
        "LOW",
        "UPSCALE_INPUT",
        "UPSCALE_FOLDER_OUTPUT",
        "esrgan",
        1.5,
        "MEDIUM",
        update(value=str(video_upscale_input)),
        "VIDEO_UPSCALE_OUTPUT",
        "esrgan",
        1.75,
        "HIGH",
        {
            **app._ui_state_defaults(),
            **state,
        },
    ]

    assert len(restored) == len(expected)
    path_like_indices = {8, 9, 26, 63, 73}
    for index, (actual, wanted) in enumerate(zip(restored, expected)):
        _assert_restored_value(actual, wanted, path_like=index in path_like_indices)

    restorable_keys = [
        "shared_prompt",
        "shared_negative",
        "edit_image",
        "edit_reference",
        "edit_output",
        "edit_steps",
        "edit_cfg",
        "edit_megapixels",
        "edit_seed",
        "edit_engine",
        "edit_live_preview",
        "edit_consistency",
        "edit_consistency_strength",
        "edit_compile",
        "edit_mrflow",
        "edit_teacache",
        "edit_depth_lock",
        "video_input",
        "video_output",
        "video_steps",
        "video_cfg",
        "video_megapixels",
        "video_seed",
        "video_engine",
        "video_compile",
        "video_mrflow",
        "video_every_n",
        "video_max_frames",
        "video_frame_dir",
        "batch_input",
        "batch_output",
        "batch_steps",
        "batch_cfg",
        "batch_megapixels",
        "batch_seed",
        "batch_engine",
        "batch_consistency",
        "batch_consistency_strength",
        "batch_compile",
        "batch_teacache",
        "batch_mrflow",
        "t2i_output",
        "t2i_width",
        "t2i_height",
        "t2i_steps",
        "t2i_cfg",
        "t2i_seed",
        "t2i_engine",
        "t2i_live_preview",
        "t2i_consistency",
        "t2i_consistency_strength",
        "t2i_compile",
        "t2i_mrflow",
        "t2i_teacache",
        "upscale_image",
        "upscale_output",
        "upscale_upscaler",
        "upscale_scale",
        "upscale_quality",
        "upscale_input_folder",
        "upscale_folder_output",
        "upscale_folder_upscaler",
        "upscale_folder_scale",
        "upscale_folder_quality",
        "video_upscale_input",
        "video_upscale_output",
        "video_upscale_upscaler",
        "video_upscale_scale",
        "video_upscale_quality",
    ]
    # model_list is refresh-only UI state, not a restorable control.
    assert set(restorable_keys) == {key for key in app._ui_state_defaults() if key != "model_list"}


def test_t2i_handler_streams_preview_then_final(monkeypatch) -> None:
    def fake_run_t2i(*args, preview_callback=None, **kwargs):
        if preview_callback is not None:
            preview_callback("preview-1")
            preview_callback("preview-2")
        return GenerationResult(image_path=Path("final.png"), status="done")

    monkeypatch.setattr(app, "run_t2i", fake_run_t2i)

    events = list(app._t2i_handler("prompt", "negative", "out", 1024, 1024, 4, 1.0, 0, "int8", False, False, False, True))

    assert events[0] == (None, "preview-1", "Rendering preview...")
    assert events[1] == (None, "preview-2", "Rendering preview...")
    assert events[-1] == ("final.png", "final.png", "done")


def test_run_depth_edit_uses_requested_base_variant(monkeypatch, tmp_path: Path) -> None:
    from comfyui_app import generation

    recorded: dict[str, object] = {}

    class FakeImage:
        def __init__(self, name: str) -> None:
            self.name = name

        def save(self, path: Path) -> None:
            Path(path).write_text(self.name, encoding="utf-8")

    class FakeClient:
        client_id = "client"

        def upload_image(self, path: Path) -> str:
            return f"uploaded:{Path(path).name}"

        def wait_until_up(self, timeout: float = 0.0) -> None:
            return None

        def queue_prompt(self, prompt_dict, client_id=None) -> str:
            recorded["diffusion_model"] = prompt_dict["1"]["inputs"]["unet_name"]
            return "prompt-id"

        def wait_for_completion(self, prompt_id, client_id=None, timeout: float = 0.0) -> None:
            return None

        def get_images(self, prompt_id: str):
            return [FakeImage("depth"), FakeImage("final")]

    def fake_depth_assets() -> tuple[str, str]:
        recorded["depth_assets_called"] = True
        return ("base.safetensors", "lora.safetensors")

    def fake_resolved_filename_map(vram_gb: float, prefer_gguf: bool, engine: str) -> dict[str, str]:
        return {"diffusion": "diff.safetensors", "text_encoder": "text.safetensors", "vae": "vae.safetensors"}

    monkeypatch.setattr(generation, "_depth_control_assets", fake_depth_assets)
    monkeypatch.setattr(generation, "_resolved_filename_map", fake_resolved_filename_map)
    monkeypatch.setattr(generation, "detect_vram", lambda: (8.0, "RTX", True))

    result = generation.run_depth_edit(tmp_path / "input.png", None, "prompt", "negative", tmp_path, client=FakeClient())
    assert recorded["depth_assets_called"] is True
    assert recorded["diffusion_model"] == "base.safetensors"
    assert result.status.startswith("Saved image to ")
    assert result.preview_path is not None
    assert result.preview_path.read_text(encoding="utf-8") == "depth"
    assert result.image_path.read_text(encoding="utf-8") == "final"


def test_run_depth_edit_forwards_teacache_to_depth_builder(monkeypatch, tmp_path: Path) -> None:
    from comfyui_app import generation

    recorded: dict[str, object] = {}

    class FakeImage:
        def __init__(self, name: str) -> None:
            self.name = name

        def save(self, path: Path) -> None:
            Path(path).write_text(self.name, encoding="utf-8")

    class FakeClient:
        client_id = "client"

        def upload_image(self, path: Path) -> str:
            return f"uploaded:{Path(path).name}"

        def wait_until_up(self, timeout: float = 0.0) -> None:
            return None

        def queue_prompt(self, prompt_dict, client_id=None) -> str:
            recorded["diffusion_model"] = prompt_dict["1"]["inputs"]["unet_name"]
            return "prompt-id"

        def wait_for_completion(self, prompt_id, client_id=None, timeout: float = 0.0) -> None:
            return None

        def get_images(self, prompt_id: str):
            return [FakeImage("depth"), FakeImage("final")]

    def fake_depth_assets() -> tuple[str, str]:
        recorded["depth_assets_called"] = True
        return ("base.safetensors", "lora.safetensors")

    def fake_resolved_filename_map(vram_gb: float, prefer_gguf: bool, engine: str) -> dict[str, str]:
        return {"diffusion": "diff.safetensors", "text_encoder": "text.safetensors", "vae": "vae.safetensors"}

    def fake_build_depth_refcontrol_edit_prompt(**kwargs):
        recorded["build_kwargs"] = kwargs
        return {"1": {"inputs": {"unet_name": kwargs["diffusion_model"]}}}

    monkeypatch.setattr(generation, "_depth_control_assets", fake_depth_assets)
    monkeypatch.setattr(generation, "_resolved_filename_map", fake_resolved_filename_map)
    monkeypatch.setattr(generation, "detect_vram", lambda: (8.0, "RTX", True))
    monkeypatch.setattr(generation, "build_depth_refcontrol_edit_prompt", fake_build_depth_refcontrol_edit_prompt)

    result = generation.run_depth_edit(tmp_path / "input.png", None, "prompt", "negative", tmp_path, use_teacache=True, client=FakeClient())
    assert recorded["depth_assets_called"] is True
    assert recorded["build_kwargs"]["use_teacache"] is True
    assert recorded["diffusion_model"] == "base.safetensors"
    assert result.status.startswith("Saved image to ")
    assert result.preview_path is not None
    assert result.preview_path.read_text(encoding="utf-8") == "depth"
    assert result.image_path.read_text(encoding="utf-8") == "final"


def test_run_edit_forwards_teacache_flag(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from comfyui_app import generation

    recorded: dict[str, object] = {}

    class FakeClient:
        client_id = "client"

        def upload_image(self, path: Path) -> str:
            return f"uploaded:{Path(path).name}"

        def wait_until_up(self, timeout: float = 0.0) -> None:
            return None

    def fake_resolved_filename_map(vram_gb: float, prefer_gguf: bool, engine: str) -> dict[str, str]:
        return {"diffusion": "diff.safetensors", "text_encoder": "text.safetensors", "vae": "vae.safetensors", "upscale": "upscale.pth"}

    def fake_build_edit_prompt(**kwargs):
        recorded["build_kwargs"] = kwargs
        return {"1": {"inputs": {"unet_name": kwargs["diffusion_model"]}}}

    def fake_run_prompt(client, prompt_dict, output_dir, output_name, timeout, preview_callback=None):
        recorded["prompt_dict"] = prompt_dict
        return GenerationResult(image_path=tmp_path / "result.png", status="ok")

    monkeypatch.setattr(generation, "_resolved_filename_map", fake_resolved_filename_map)
    monkeypatch.setattr(generation, "detect_vram", lambda: (8.0, "RTX", True))
    monkeypatch.setattr(generation, "select_tier", lambda vram_gb: SimpleNamespace(use_tiled_decode=True))
    monkeypatch.setattr(generation, "build_edit_prompt", fake_build_edit_prompt)
    monkeypatch.setattr(generation, "_run_prompt", fake_run_prompt)

    result = generation.run_edit(tmp_path / "input.png", "prompt", "negative", tmp_path, use_teacache=True, client=FakeClient())

    assert recorded["build_kwargs"]["use_teacache"] is True
    assert result.status == "ok"
