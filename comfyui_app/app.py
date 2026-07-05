from __future__ import annotations

import logging
import os
import shutil
import threading
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from queue import Queue

try:
    from starlette.exceptions import StarletteDeprecationWarning
except Exception:  # pragma: no cover - optional dependency
    StarletteDeprecationWarning = None  # type: ignore[assignment]

if StarletteDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*resume_download.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*resume_download.*")

try:
    import gradio as gr
except Exception:  # pragma: no cover - optional dependency
    gr = None  # type: ignore[assignment]

from comfyui_app.batch import CANCEL_EVENT, clear_cancel, iter_process_folder, process_folder, request_cancel
from comfyui_app.comfy_client import ComfyClient
from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT, DEFAULT_OUTPUT_DIR
from comfyui_app.generation import GenerationResult, run_depth_edit, run_edit, run_t2i, run_upscale
from comfyui_app.model_manager import delete_models as delete_installed_models
from comfyui_app.model_manager import find_removable_models, list_installed_models, remove_unused_models
from comfyui_app.model_resolver import ModelResolverError, load_resolved_manifest
from comfyui_app.ui_utils import pick_directory
from comfyui_app.vram import detect_vram, select_tier
from comfyui_app.video_frames import extract_frames, frames_to_video, probe_video_metadata

logger = logging.getLogger(__name__)

ENGINE_CHOICES = [
    ("INT8 (fastest on Ampere - default)", "int8"),
    ("Nunchaku INT4 (experimental - faster, needs extra install)", "nunchaku_int4"),
]
UPSCALER_CHOICES = [
    ("NVIDIA RTX VSR (experimental)", "rtx"),
    ("Real-ESRGAN x2+", "esrgan"),
]
QUALITY_CHOICES = [("LOW", "LOW"), ("MEDIUM", "MEDIUM"), ("HIGH", "HIGH"), ("ULTRA", "ULTRA")]
VIDEO_EXTS = [".mp4", ".mov"]
DEFAULT_ENGINE_VALUE = "int8"
UI_STATE_STORAGE_KEY = "comfyui_local_image_app_ui_state_v1"
UI_STATE_SECRET = "comfyui_local_image_app_ui_state_secret_v1"


def _ui_state_defaults() -> dict[str, object]:
    return {
        "shared_prompt": "",
        "shared_negative": "",
        "edit_image": None,
        "edit_reference": None,
        "edit_output": str(DEFAULT_OUTPUT_DIR),
        "edit_steps": 4,
        "edit_cfg": 1.0,
        "edit_megapixels": 1.0,
        "edit_seed": 0,
        "edit_engine": DEFAULT_ENGINE_VALUE,
        "edit_live_preview": False,
        "edit_consistency": False,
        "edit_consistency_strength": 1.0,
        "edit_compile": False,
        "edit_mrflow": False,
        "edit_teacache": False,
        "edit_depth_lock": False,
        "video_input": None,
        "video_output": str(DEFAULT_OUTPUT_DIR),
        "video_steps": 4,
        "video_cfg": 1.0,
        "video_megapixels": 1.0,
        "video_seed": 0,
        "video_engine": DEFAULT_ENGINE_VALUE,
        "video_compile": False,
        "video_mrflow": False,
        "video_every_n": 1,
        "video_max_frames": 0,
        "video_frame_dir": str(DEFAULT_OUTPUT_DIR),
        "batch_input": str(DEFAULT_OUTPUT_DIR),
        "batch_output": str(DEFAULT_OUTPUT_DIR),
        "batch_steps": 4,
        "batch_cfg": 1.0,
        "batch_megapixels": 1.0,
        "batch_seed": 0,
        "batch_engine": DEFAULT_ENGINE_VALUE,
        "batch_consistency": False,
        "batch_consistency_strength": 1.0,
        "batch_compile": False,
        "batch_teacache": False,
        "batch_mrflow": False,
        "t2i_output": str(DEFAULT_OUTPUT_DIR),
        "t2i_width": 1024,
        "t2i_height": 1024,
        "t2i_steps": 4,
        "t2i_cfg": 1.0,
        "t2i_seed": 0,
        "t2i_engine": DEFAULT_ENGINE_VALUE,
        "t2i_live_preview": False,
        "t2i_consistency": False,
        "t2i_consistency_strength": 1.0,
        "t2i_compile": False,
        "t2i_mrflow": False,
        "t2i_teacache": False,
        "upscale_image": None,
        "upscale_output": str(DEFAULT_OUTPUT_DIR),
        "upscale_upscaler": "rtx",
        "upscale_scale": 2.0,
        "upscale_quality": "ULTRA",
        "upscale_input_folder": str(DEFAULT_OUTPUT_DIR),
        "upscale_folder_output": str(DEFAULT_OUTPUT_DIR),
        "upscale_folder_upscaler": "rtx",
        "upscale_folder_scale": 2.0,
        "upscale_folder_quality": "ULTRA",
        "video_upscale_input": None,
        "video_upscale_output": str(DEFAULT_OUTPUT_DIR),
        "video_upscale_upscaler": "rtx",
        "video_upscale_scale": 2.0,
        "video_upscale_quality": "ULTRA",
        "model_list": [],
    }


def _ui_state(state: object | None) -> dict[str, object]:
    merged = _ui_state_defaults()
    if isinstance(state, dict):
        for key in merged:
            if key in state:
                merged[key] = state[key]
    return merged


def _ui_state_update(state: object | None, key: str, value: object) -> dict[str, object]:
    updated = _ui_state(state)
    updated[key] = value
    return updated


def _ui_state_path(state: object | None, key: str) -> str | None:
    value = _ui_state(state).get(key)
    if isinstance(value, dict):
        try:
            value = _component_path(value)
        except Exception:
            return None
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value:
        return None
    if not Path(value).exists():
        return None
    return value


def _ui_state_choice(state: object | None, key: str, choices: set[str], default: str) -> str:
    value = _ui_state(state).get(key, default)
    return value if isinstance(value, str) and value in choices else default


def _ui_state_text(state: object | None, key: str) -> str:
    value = _ui_state(state).get(key, "")
    return value if isinstance(value, str) else str(value or "")


def _restore_ui_state(state: object | None) -> tuple[object, ...]:
    ui_state = _ui_state(state)
    edit_depth_lock = bool(ui_state["edit_depth_lock"])
    edit_live_preview = bool(ui_state["edit_live_preview"])
    edit_consistency = bool(ui_state["edit_consistency"])
    batch_consistency = bool(ui_state["batch_consistency"])
    t2i_live_preview = bool(ui_state["t2i_live_preview"])
    t2i_consistency = bool(ui_state["t2i_consistency"])
    return (
        ui_state["shared_prompt"],
        ui_state["shared_prompt"],
        ui_state["shared_prompt"],
        ui_state["shared_prompt"],
        ui_state["shared_negative"],
        ui_state["shared_negative"],
        ui_state["shared_negative"],
        ui_state["shared_negative"],
        gr.update(value=_ui_state_path(ui_state, "edit_image")),
        gr.update(value=_ui_state_path(ui_state, "edit_reference"), visible=edit_depth_lock),
        ui_state["edit_output"],
        ui_state["edit_steps"],
        ui_state["edit_cfg"],
        ui_state["edit_megapixels"],
        ui_state["edit_seed"],
        _ui_state_choice(ui_state, "edit_engine", {"int8", "nunchaku_int4"}, DEFAULT_ENGINE_VALUE),
        ui_state["edit_live_preview"],
        gr.update(visible=edit_live_preview),
        ui_state["edit_consistency"],
        gr.update(value=ui_state["edit_consistency_strength"], visible=edit_consistency),
        ui_state["edit_compile"],
        ui_state["edit_mrflow"],
        ui_state["edit_teacache"],
        ui_state["edit_depth_lock"],
        gr.update(visible=edit_depth_lock),
        gr.update(visible=edit_depth_lock),
        gr.update(value=_ui_state_path(ui_state, "video_input")),
        ui_state["video_output"],
        ui_state["video_steps"],
        ui_state["video_cfg"],
        ui_state["video_megapixels"],
        ui_state["video_seed"],
        _ui_state_choice(ui_state, "video_engine", {"int8", "nunchaku_int4"}, DEFAULT_ENGINE_VALUE),
        ui_state["video_compile"],
        ui_state["video_mrflow"],
        ui_state["video_every_n"],
        ui_state["video_max_frames"],
        ui_state["video_frame_dir"],
        ui_state["batch_input"],
        ui_state["batch_output"],
        ui_state["batch_steps"],
        ui_state["batch_cfg"],
        ui_state["batch_megapixels"],
        ui_state["batch_seed"],
        _ui_state_choice(ui_state, "batch_engine", {"int8", "nunchaku_int4"}, DEFAULT_ENGINE_VALUE),
        ui_state["batch_consistency"],
        gr.update(value=ui_state["batch_consistency_strength"], visible=batch_consistency),
        ui_state["batch_compile"],
        ui_state["batch_teacache"],
        ui_state["batch_mrflow"],
        ui_state["t2i_output"],
        ui_state["t2i_width"],
        ui_state["t2i_height"],
        ui_state["t2i_steps"],
        ui_state["t2i_cfg"],
        ui_state["t2i_seed"],
        _ui_state_choice(ui_state, "t2i_engine", {"int8", "nunchaku_int4"}, DEFAULT_ENGINE_VALUE),
        ui_state["t2i_live_preview"],
        ui_state["t2i_consistency"],
        gr.update(value=ui_state["t2i_consistency_strength"], visible=t2i_consistency),
        ui_state["t2i_compile"],
        ui_state["t2i_mrflow"],
        ui_state["t2i_teacache"],
        gr.update(value=_ui_state_path(ui_state, "upscale_image")),
        ui_state["upscale_output"],
        _ui_state_choice(ui_state, "upscale_upscaler", {"rtx", "esrgan"}, "rtx"),
        ui_state["upscale_scale"],
        _ui_state_choice(ui_state, "upscale_quality", {"LOW", "MEDIUM", "HIGH", "ULTRA"}, "ULTRA"),
        ui_state["upscale_input_folder"],
        ui_state["upscale_folder_output"],
        _ui_state_choice(ui_state, "upscale_folder_upscaler", {"rtx", "esrgan"}, "rtx"),
        ui_state["upscale_folder_scale"],
        _ui_state_choice(ui_state, "upscale_folder_quality", {"LOW", "MEDIUM", "HIGH", "ULTRA"}, "ULTRA"),
        gr.update(value=_ui_state_path(ui_state, "video_upscale_input")),
        ui_state["video_upscale_output"],
        _ui_state_choice(ui_state, "video_upscale_upscaler", {"rtx", "esrgan"}, "rtx"),
        ui_state["video_upscale_scale"],
        _ui_state_choice(ui_state, "video_upscale_quality", {"LOW", "MEDIUM", "HIGH", "ULTRA"}, "ULTRA"),
        ui_state,
    )


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, ModelResolverError):
        return exc.message
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _as_gr_image(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _component_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "filepath", "tempfile"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return Path(item)
    return Path(str(value))


def _optional_component_path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    try:
        return _component_path(value)
    except Exception:
        return None


def _status_markdown() -> str:
    manifest = load_resolved_manifest()
    if not isinstance(manifest, dict):
        return "Ready. No resolved model manifest found yet."
    models = manifest.get("models")
    engine = str(manifest.get("engine", "int8"))
    if engine == "default":
        engine = "int8"
    if not isinstance(models, dict):
        return "Ready. The resolved model manifest is present, but it is incomplete."
    lines = [f"Ready. Current manifest engine: `{engine}`."]
    for key in ("diffusion", "text_encoder", "vae", "upscale"):
        entry = models.get(key)
        if isinstance(entry, dict):
            filename = entry.get("local_filename", "")
            lines.append(f"- {key}: `{filename}`")
    return "\n".join(lines)


def refresh_status() -> str:
    try:
        vram_gb, device_name, cuda_available = detect_vram()
        tier = select_tier(vram_gb)
        device_line = f"Detected {device_name} ({vram_gb:.1f} GB VRAM)" if cuda_available else "No CUDA GPU detected"
        return f"**{device_line}**\n\nUsing the **{tier.label}** setup.\n\n{_status_markdown()}"
    except Exception as exc:
        return _friendly_error(exc)


def _stop_current_job() -> str:
    request_cancel()
    try:
        ComfyClient(COMFYUI_HOST, COMFYUI_PORT).interrupt()
    except Exception:
        logger.debug("Interrupt request failed", exc_info=True)
    return "Stop requested."


def _model_manager_payload() -> tuple[object, str, str]:
    data = list_installed_models()
    choices = [(entry["label"], entry["path"]) for entry in data["entries"]]
    total_line = f"**Total installed model files:** {data['count']}  \n**Disk usage:** {data['total']}"
    status = f"Found {data['count']} model files."
    return gr.update(choices=choices, value=[]), total_line, status


def _model_manager_refresh() -> tuple[object, str, str]:
    try:
        return _model_manager_payload()
    except Exception as exc:
        return gr.update(choices=[], value=[]), "Unable to read installed models.", _friendly_error(exc)


def _model_manager_delete(selected_paths: list[str] | None) -> tuple[object, str, str]:
    try:
        if not selected_paths:
            refreshed = _model_manager_payload()
            return refreshed[0], refreshed[1], "No models selected."
        data = delete_installed_models(selected_paths)
        choices = [(entry["label"], entry["path"]) for entry in data["entries"]]
        total_line = f"**Total installed model files:** {data['count']}  \n**Disk usage:** {data['total']}"
        status = f"Deleted {data['freed']} and refreshed the list."
        return gr.update(choices=choices, value=[]), total_line, status
    except Exception as exc:
        refreshed = _model_manager_payload()
        return refreshed[0], refreshed[1], _friendly_error(exc)


def _model_manager_cleanup_preview() -> tuple[object, str, object, list[str]]:
    try:
        data = find_removable_models()
        if not data["entries"]:
            return (
                "**No unused or duplicate models were found.**",
                "Nothing to remove.",
                gr.update(visible=False),
                [],
            )
        lines = ["**Removable models:**"]
        for entry in data["entries"]:
            lines.append(f"- `{entry['reason']}`: `{entry['category']}/{entry['filename']}` ({entry['size']})")
        lines.append("")
        lines.append(f"**Reclaimable space:** {data['total']} across {data['count']} files")
        return "\n".join(lines), "Review the list, then confirm removal.", gr.update(visible=True), [
            str(entry["path"]) for entry in data["entries"] if isinstance(entry.get("path"), str)
        ]
    except Exception as exc:
        return "**Unable to scan removable models.**", _friendly_error(exc), gr.update(visible=False), []


def _model_manager_cleanup_confirm(removable_paths: list[str] | None) -> tuple[object, str, object, str, object, list[str]]:
    try:
        if not removable_paths:
            refreshed = _model_manager_payload()
            return refreshed[0], refreshed[1], "**No removable models selected.**", "Nothing to remove.", gr.update(visible=False), []
        data = remove_unused_models(removable_paths)
        refreshed = _model_manager_payload()
        return (
            refreshed[0],
            refreshed[1],
            f"**Removed {data['freed']} of unused / duplicate models.**",
            f"Deleted {data['freed']} and refreshed the list.",
            gr.update(visible=False),
            [],
        )
    except Exception as exc:
        refreshed = _model_manager_payload()
        return refreshed[0], refreshed[1], "**Cleanup failed.**", _friendly_error(exc), gr.update(visible=False), []


def _edit_handler(
    input_image: object,
    reference_image: object,
    prompt: str,
    negative: str,
    output_dir: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
    use_teacache: bool,
    depth_lock: bool,
    live_preview: bool,
    consistency: bool = False,
    consistency_strength: float = 1.0,
) -> object:
    try:
        input_path = _component_path(input_image)
        reference_path = _optional_component_path(reference_image)
        if depth_lock:
            result = run_depth_edit(
                input_path,
                reference_path,
                prompt,
                negative,
                output_dir,
                seed=int(seed),
                megapixels=float(megapixels),
                use_teacache=bool(use_teacache),
            )
            yield str(result.image_path), None, _as_gr_image(result.preview_path), result.status
            return
        if not live_preview or bool(mrflow):
            result = run_edit(
                input_path,
                prompt,
                negative,
                output_dir,
                steps=int(steps),
                cfg=float(cfg),
                seed=int(seed),
                megapixels=float(megapixels),
                engine=engine,
                use_torch_compile=bool(use_torch_compile),
                use_teacache=bool(use_teacache),
                use_consistency_lora=bool(consistency),
                consistency_lora_name=None,
                consistency_lora_strength=float(consistency_strength),
                mrflow=bool(mrflow),
            )
            yield str(result.image_path), None, None, result.status
            return

        queue: Queue[object] = Queue()
        sentinel = object()
        state: dict[str, object] = {}

        def preview_callback(image: object) -> None:
            queue.put(image)

        def worker() -> None:
            try:
                state["result"] = run_edit(
                    input_path,
                    prompt,
                    negative,
                    output_dir,
                    steps=int(steps),
                    cfg=float(cfg),
                    seed=int(seed),
                    megapixels=float(megapixels),
                    engine=engine,
                    use_torch_compile=bool(use_torch_compile),
                    use_teacache=bool(use_teacache),
                    use_consistency_lora=bool(consistency),
                    consistency_lora_name=None,
                    consistency_lora_strength=float(consistency_strength),
                    mrflow=bool(mrflow),
                    preview_callback=preview_callback,
                )
            except Exception as exc:
                state["exc"] = exc
            finally:
                queue.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        last_preview: object | None = None
        while True:
            item = queue.get()
            if item is sentinel:
                break
            last_preview = item
            yield None, last_preview, None, "Rendering preview..."
        thread.join()
        if "exc" in state:
            yield None, None, None, _friendly_error(state["exc"])  # type: ignore[arg-type]
            return
        result = state["result"]  # type: ignore[assignment]
        yield str(result.image_path), str(result.image_path), None, result.status
    except Exception as exc:
        yield None, None, None, _friendly_error(exc)


def _t2i_handler(
    prompt: str,
    negative: str,
    output_dir: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
    use_teacache: bool,
    live_preview: bool,
    consistency: bool = False,
    consistency_strength: float = 1.0,
) -> object:
    try:
        if not live_preview:
            result = run_t2i(
                prompt=prompt,
                negative=negative,
                output_dir=output_dir,
                width=int(width),
                height=int(height),
                steps=int(steps),
                cfg=float(cfg),
                seed=int(seed),
                engine=engine,
                use_teacache=bool(use_teacache),
                use_torch_compile=bool(use_torch_compile),
                use_consistency_lora=bool(consistency),
                consistency_lora_name=None,
                consistency_lora_strength=float(consistency_strength),
                mrflow=bool(mrflow),
            )
            yield str(result.image_path), None, result.status
            return

        queue: Queue[object] = Queue()
        sentinel = object()
        state: dict[str, object] = {}

        def preview_callback(image: object) -> None:
            queue.put(image)

        def worker() -> None:
            try:
                state["result"] = run_t2i(
                    prompt=prompt,
                    negative=negative,
                    output_dir=output_dir,
                    width=int(width),
                    height=int(height),
                    steps=int(steps),
                    cfg=float(cfg),
                    seed=int(seed),
                    engine=engine,
                    use_teacache=bool(use_teacache),
                    use_torch_compile=bool(use_torch_compile),
                    use_consistency_lora=bool(consistency),
                    consistency_lora_name=None,
                    consistency_lora_strength=float(consistency_strength),
                    mrflow=bool(mrflow),
                    preview_callback=preview_callback,
                )
            except Exception as exc:
                state["exc"] = exc
            finally:
                queue.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        last_preview: object | None = None
        while True:
            item = queue.get()
            if item is sentinel:
                break
            last_preview = item
            yield None, last_preview, "Rendering preview..."
        thread.join()
        if "exc" in state:
            yield None, None, _friendly_error(state["exc"])  # type: ignore[arg-type]
            return
        result = state["result"]  # type: ignore[assignment]
        yield str(result.image_path), str(result.image_path), result.status
    except Exception as exc:
        yield None, None, _friendly_error(exc)


def _upscale_handler(
    input_image: object,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
) -> tuple[str | None, str]:
    try:
        result = run_upscale(
            _component_path(input_image),
            output_dir,
            upscaler=upscaler,
            scale=float(scale),
            quality=quality,
        )
        return str(result.image_path), result.status
    except Exception as exc:
        return None, _friendly_error(exc)


def _yield_folder_updates(summary_iter):
    last_state: tuple[object, ...] | None = None
    for summary in summary_iter:
        results = tuple(summary.get("results", []))
        failures = tuple(summary.get("failures", []))
        state = (summary.get("count"), results, failures, summary.get("message"), summary.get("output_dir"))
        if state == last_state:
            continue
        last_state = state
        yield summary.get("message", ""), list(results)


def _batch_folder_stream(
    input_dir: str,
    output_dir: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
    use_teacache: bool,
    use_consistency_lora: bool,
    consistency_lora_strength: float,
):
    client = ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    client_ready = False

    def ensure_client_ready() -> None:
        nonlocal client_ready
        if not client_ready:
            client.wait_until_up()
            client_ready = True

    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        ensure_client_ready()
        return run_edit(
            image_path,
            prompt_text,
            negative_text,
            run_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            use_teacache=bool(use_teacache),
            use_consistency_lora=bool(use_consistency_lora),
            consistency_lora_name=None,
            consistency_lora_strength=float(consistency_lora_strength),
            mrflow=bool(mrflow),
            client=client,
        )

    yield from _yield_folder_updates(iter_process_folder(input_dir, output_dir, prompt, negative, gen_fn))


def _upscale_folder_stream(
    input_dir: str,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
):
    client = ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    client_ready = False

    def ensure_client_ready() -> None:
        nonlocal client_ready
        if not client_ready:
            client.wait_until_up()
            client_ready = True

    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        ensure_client_ready()
        return run_upscale(image_path, run_dir, upscaler=upscaler, scale=float(scale), quality=quality, client=client)

    yield from _yield_folder_updates(iter_process_folder(input_dir, output_dir, "", "", gen_fn))


def _extract_frames_handler(video_file: object, output_dir: str, every_n: int, max_frames: int) -> tuple[str, str]:
    try:
        video_path = _component_path(video_file)
        run_dir = Path(output_dir) / f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved = extract_frames(video_path, run_dir, every_n=int(every_n), max_frames=int(max_frames) or None)
        return str(run_dir), f"Extracted {len(saved)} frames to {run_dir}."
    except Exception as exc:
        return output_dir, _friendly_error(exc)


def _edit_frames_handler(
    input_dir: str,
    output_dir: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    megapixels: float,
    seed: int,
    engine: str,
    use_torch_compile: bool,
    mrflow: bool,
) -> tuple[str]:
    client = ComfyClient(COMFYUI_HOST, COMFYUI_PORT)
    client_ready = False

    def ensure_client_ready() -> None:
        nonlocal client_ready
        if not client_ready:
            client.wait_until_up()
            client_ready = True

    def gen_fn(image_path: Path, prompt_text: str, negative_text: str, run_dir: Path) -> GenerationResult:
        ensure_client_ready()
        return run_edit(
            image_path,
            prompt_text,
            negative_text,
            run_dir,
            steps=int(steps),
            cfg=float(cfg),
            seed=int(seed),
            megapixels=float(megapixels),
            engine=engine,
            use_torch_compile=bool(use_torch_compile),
            mrflow=bool(mrflow),
            client=client,
        )

    try:
        summary = process_folder(input_dir, output_dir, prompt, negative, gen_fn)
        return (summary["message"],)
    except Exception as exc:
        return (_friendly_error(exc),)


def _validate_video_file(video_file: object) -> Path:
    video_path = _component_path(video_file)
    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ModelResolverError("Video Upscale only accepts .mp4 and .mov input files.")
    return video_path


def _video_upscale_handler(
    video_file: object,
    output_dir: str,
    upscaler: str,
    scale: float,
    quality: str,
) -> tuple[str | None, str]:
    try:
        clear_cancel()
        source_path = _validate_video_file(video_file)
        metadata = probe_video_metadata(source_path)
        fps = float(metadata.get("fps") or 0.0) or 24.0
        source_width = int(metadata.get("width") or 0)
        source_height = int(metadata.get("height") or 0)
        has_audio = bool(metadata.get("has_audio"))
        target_width = max(1, int(round(source_width * float(scale)))) if source_width > 0 else None
        target_height = max(1, int(round(source_height * float(scale)))) if source_height > 0 else None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{source_path.stem}_upscaled.mp4"
        with TemporaryDirectory(dir=str(output_root)) as work_dir:
            work_root = Path(work_dir)
            frames_dir = work_root / "frames"
            upscaled_dir = work_root / "upscaled"
            extracted_frames = extract_frames(source_path, frames_dir)
            if CANCEL_EVENT.is_set():
                return None, "Cancelled."
            upscaled_dir.mkdir(parents=True, exist_ok=True)
            for index, frame_path in enumerate(extracted_frames):
                if CANCEL_EVENT.is_set():
                    return None, f"Cancelled after {index} frames."
                result = run_upscale(
                    frame_path,
                    upscaled_dir,
                    upscaler=upscaler,
                    scale=float(scale),
                    quality=quality,
                    target_width=target_width,
                    target_height=target_height,
                )
                destination = upscaled_dir / f"frame_{index:06d}.png"
                shutil.copy2(result.image_path, destination)
            if CANCEL_EVENT.is_set():
                return None, "Cancelled."
            final_path = frames_to_video(upscaled_dir, output_path, fps, audio_source=source_path if has_audio else None)
            return str(final_path), f"Saved video to {final_path}."
    except Exception as exc:
        return None, _friendly_error(exc)


def build_app() -> "gr.Blocks":
    if gr is None:
        raise RuntimeError("Gradio is not installed.")

    with gr.Blocks(title="ComfyUI Local Image App") as demo:
        gr.Markdown("# ComfyUI Local Image App")
        status_box = gr.Markdown(_status_markdown())
        refresh_button = gr.Button("Refresh status")
        refresh_button.click(fn=refresh_status, outputs=status_box)
        ui_state = gr.BrowserState(
            _ui_state_defaults(),
            storage_key=UI_STATE_STORAGE_KEY,
            secret=UI_STATE_SECRET,
        )

        def directory_field(label: str, value: str, *, state_key: str | None = None) -> tuple[object, object]:
            with gr.Row():
                textbox = gr.Textbox(label=label, value=value, scale=6)
                browse_button = gr.Button("Browse...", scale=1)
            if state_key is None:
                browse_button.click(fn=pick_directory, inputs=[textbox], outputs=[textbox])
            else:
                textbox.input(
                    fn=lambda state, text: _ui_state_update(state, state_key, text or ""),
                    inputs=[ui_state, textbox],
                    outputs=[ui_state],
                )
                def _browse_and_store(current_value: str, state: object) -> tuple[str, object]:
                    selected = pick_directory(current_value)
                    return selected, _ui_state_update(state, state_key, selected)

                browse_button.click(fn=_browse_and_store, inputs=[textbox, ui_state], outputs=[textbox, ui_state])
            return textbox, browse_button

        with gr.Tab("Image Edit"):
            with gr.Row():
                with gr.Column():
                    edit_image = gr.Image(label="Input image", type="filepath")
                    edit_reference = gr.Image(label="Identity image (optional)", type="filepath", visible=False)
                    edit_depth_note = gr.Markdown(
                        "With Pose/Shape lock, the output is locked to the input image's pose/shape (depth is auto-extracted from it). Leave Identity empty to keep the input's own identity, or add an identity image to borrow a different subject's identity while keeping the input's pose/shape.",
                        visible=False,
                    )
                    edit_depth_preview = gr.Image(label="Depth map preview", visible=False)
                    edit_prompt = gr.Textbox(label="Prompt", lines=4)
                    edit_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    edit_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="edit_output")
                    edit_steps = gr.Number(label="Steps", value=4, precision=0)
                    edit_cfg = gr.Number(label="Guidance", value=1.0)
                    edit_megapixels = gr.Number(label="Megapixels", value=1.0)
                    edit_seed = gr.Number(label="Seed", value=0, precision=0)
                    edit_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value=DEFAULT_ENGINE_VALUE)
                    edit_live_preview = gr.Checkbox(label="Show live preview", value=False)
                    edit_live_preview_image = gr.Image(label="Live preview", visible=False)
                    edit_consistency = gr.Checkbox(
                        label="Consistency LoRA (INT8 FLUX.2 Klein 4B)",
                        value=False,
                        info="Auto-downloads and applies the FLUX.2 Klein 4B consistency LoRA on first use.",
                    )
                    edit_consistency_strength = gr.Slider(
                        label="Consistency LoRA strength",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        visible=False,
                    )
                    edit_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    edit_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_teacache = gr.Checkbox(
                        label="Experimental: TeaCache speedup (multi-step flows only)",
                        value=False,
                    )
                    edit_depth_lock = gr.Checkbox(
                        label="Pose/Shape lock (depth reference) — experimental, slower (~20 steps, base model)",
                        value=False,
                    )
                    edit_button = gr.Button("Generate")
                    edit_stop = gr.Button("Stop")
                    edit_result = gr.Image(label="Result")
                    edit_status = gr.Textbox(label="Status")
            edit_run = edit_button.click(
                fn=_edit_handler,
                inputs=[edit_image, edit_reference, edit_prompt, edit_negative, edit_output, edit_steps, edit_cfg, edit_megapixels, edit_seed, edit_engine, edit_compile, edit_mrflow, edit_teacache, edit_depth_lock, edit_live_preview, edit_consistency, edit_consistency_strength],
                outputs=[edit_result, edit_live_preview_image, edit_depth_preview, edit_status],
            )
            edit_depth_lock.change(
                fn=lambda enabled: (gr.update(visible=bool(enabled)), gr.update(visible=bool(enabled)), gr.update(visible=bool(enabled))),
                inputs=[edit_depth_lock],
                outputs=[edit_reference, edit_depth_note, edit_depth_preview],
            )
            edit_live_preview.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[edit_live_preview],
                outputs=[edit_live_preview_image],
            )
            edit_consistency.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[edit_consistency],
                outputs=[edit_consistency_strength],
            )
            edit_stop.click(fn=_stop_current_job, outputs=edit_status, cancels=[edit_run])

        with gr.Tab("Video to Frames"):
            frame_state = gr.State("")
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video")
                    every_n = gr.Number(label="Every Nth frame", value=1, precision=0)
                    max_frames = gr.Number(label="Max frames", value=0, precision=0)
                    video_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="video_output")
                    extract_button = gr.Button("Extract frames")
                    frame_status = gr.Textbox(label="Status")
                with gr.Column():
                    frame_dir = gr.Textbox(label="Extracted frames folder", value=str(DEFAULT_OUTPUT_DIR))
                    video_prompt = gr.Textbox(label="Prompt", lines=4)
                    video_negative = gr.Textbox(label="Negative prompt", lines=3)
                    video_steps = gr.Number(label="Steps", value=4, precision=0)
                    video_cfg = gr.Number(label="Guidance", value=1.0)
                    video_megapixels = gr.Number(label="Megapixels", value=1.0)
                    video_seed = gr.Number(label="Seed", value=0, precision=0)
                    video_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value=DEFAULT_ENGINE_VALUE)
                    video_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    video_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    edit_frames_button = gr.Button("Edit all frames")
                    edit_frames_stop = gr.Button("Stop")
            extract_button.click(
                fn=_extract_frames_handler,
                inputs=[video_input, video_output, every_n, max_frames],
                outputs=[frame_dir, frame_status],
            )
            edit_frames_evt = edit_frames_button.click(
                fn=_edit_frames_handler,
                inputs=[frame_dir, video_output, video_prompt, video_negative, video_steps, video_cfg, video_megapixels, video_seed, video_engine, video_compile, video_mrflow],
                outputs=frame_status,
            )
            edit_frames_stop.click(fn=_stop_current_job, outputs=frame_status, cancels=[edit_frames_evt])

        with gr.Tab("Batch Folder"):
            with gr.Row():
                with gr.Column():
                    batch_input, _ = directory_field("Input folder", str(DEFAULT_OUTPUT_DIR), state_key="batch_input")
                    batch_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="batch_output")
                    batch_prompt = gr.Textbox(label="Prompt", lines=4)
                    batch_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    batch_steps = gr.Number(label="Steps", value=4, precision=0)
                    batch_cfg = gr.Number(label="Guidance", value=1.0)
                    batch_megapixels = gr.Number(label="Megapixels", value=1.0)
                    batch_seed = gr.Number(label="Seed", value=0, precision=0)
                    batch_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value=DEFAULT_ENGINE_VALUE)
                    batch_consistency = gr.Checkbox(
                        label="Consistency LoRA (INT8 FLUX.2 Klein 4B)",
                        value=False,
                        info="Auto-downloads and applies the FLUX.2 Klein 4B consistency LoRA on first use.",
                    )
                    batch_consistency_strength = gr.Slider(
                        label="Consistency LoRA strength",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        visible=False,
                    )
                    batch_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    batch_teacache = gr.Checkbox(
                        label="Experimental: TeaCache speedup (multi-step flows only)",
                        value=False,
                    )
                    batch_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    batch_button = gr.Button("Process folder")
                    batch_stop = gr.Button("Stop")
                    batch_status = gr.Textbox(label="Status")
                    batch_gallery = gr.Gallery(label="Results", columns=3, height=240)
            batch_evt = batch_button.click(
                fn=_batch_folder_stream,
                inputs=[batch_input, batch_output, batch_prompt, batch_negative, batch_steps, batch_cfg, batch_megapixels, batch_seed, batch_engine, batch_compile, batch_mrflow, batch_teacache, batch_consistency, batch_consistency_strength],
                outputs=[batch_status, batch_gallery],
            )
            batch_consistency.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[batch_consistency],
                outputs=[batch_consistency_strength],
            )
            batch_stop.click(fn=_stop_current_job, outputs=batch_status, cancels=[batch_evt])

        with gr.Tab("Text-to-Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Prompt", lines=4)
                    t2i_negative = gr.Textbox(label="Negative prompt", lines=3)
                with gr.Column():
                    t2i_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="t2i_output")
                    t2i_width = gr.Number(label="Width", value=1024, precision=0)
                    t2i_height = gr.Number(label="Height", value=1024, precision=0)
                    t2i_steps = gr.Number(label="Steps", value=4, precision=0)
                    t2i_cfg = gr.Number(label="Guidance", value=1.0)
                    t2i_seed = gr.Number(label="Seed", value=0, precision=0)
                    t2i_engine = gr.Dropdown(label="Engine", choices=ENGINE_CHOICES, value=DEFAULT_ENGINE_VALUE)
                    t2i_live_preview = gr.Checkbox(label="Show live preview", value=False)
                    t2i_live_preview_image = gr.Image(label="Live preview", visible=False)
                    t2i_consistency = gr.Checkbox(
                        label="Consistency LoRA (INT8 FLUX.2 Klein 4B)",
                        value=False,
                        info="Auto-downloads and applies the FLUX.2 Klein 4B consistency LoRA on first use.",
                    )
                    t2i_consistency_strength = gr.Slider(
                        label="Consistency LoRA strength",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        visible=False,
                    )
                    t2i_compile = gr.Checkbox(
                        label="torch.compile (requires Triton from experimental speedups; limited gain on Ampere; faster after warmup, slower first run, recompiles on resolution change)",
                        value=False,
                    )
                    t2i_mrflow = gr.Checkbox(
                        label="MrFlow staged (experimental - faster; low-res generate + upscale + refine)",
                        value=False,
                    )
                    t2i_teacache = gr.Checkbox(
                        label="Experimental: TeaCache speedup (multi-step flows only)",
                        value=False,
                    )
                    t2i_button = gr.Button("Generate")
                    t2i_stop = gr.Button("Stop")
                    t2i_result = gr.Image(label="Result")
                    t2i_status = gr.Textbox(label="Status")
            t2i_evt = t2i_button.click(
                fn=_t2i_handler,
                inputs=[t2i_prompt, t2i_negative, t2i_output, t2i_width, t2i_height, t2i_steps, t2i_cfg, t2i_seed, t2i_engine, t2i_compile, t2i_mrflow, t2i_teacache, t2i_live_preview, t2i_consistency, t2i_consistency_strength],
                outputs=[t2i_result, t2i_live_preview_image, t2i_status],
            )
            t2i_live_preview.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[t2i_live_preview],
                outputs=[t2i_live_preview_image],
            )
            t2i_consistency.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[t2i_consistency],
                outputs=[t2i_consistency_strength],
            )
            t2i_stop.click(fn=_stop_current_job, outputs=t2i_status, cancels=[t2i_evt])

        with gr.Tab("Upscale"):
            with gr.Tabs():
                with gr.Tab("Single image"):
                    with gr.Row():
                        with gr.Column():
                            upscale_image = gr.Image(label="Image", type="filepath")
                            upscale_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="upscale_output")
                        with gr.Column():
                            upscale_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                            upscale_scale = gr.Number(label="Scale", value=2.0, precision=2)
                            upscale_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                            upscale_button = gr.Button("Generate")
                            upscale_stop = gr.Button("Stop")
                            upscale_result = gr.Image(label="Result")
                            upscale_status = gr.Textbox(label="Status")
                    upscale_evt = upscale_button.click(
                        fn=_upscale_handler,
                        inputs=[upscale_image, upscale_output, upscale_upscaler, upscale_scale, upscale_quality],
                        outputs=[upscale_result, upscale_status],
                    )
                    upscale_stop.click(fn=_stop_current_job, outputs=upscale_status, cancels=[upscale_evt])

                with gr.Tab("Folder"):
                    with gr.Row():
                        with gr.Column():
                            upscale_input_folder, _ = directory_field("Input folder", str(DEFAULT_OUTPUT_DIR), state_key="upscale_input_folder")
                            upscale_folder_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="upscale_folder_output")
                        with gr.Column():
                            upscale_folder_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                            upscale_folder_scale = gr.Number(label="Scale", value=2.0, precision=2)
                            upscale_folder_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                            upscale_folder_button = gr.Button("Process folder")
                            upscale_folder_stop = gr.Button("Stop")
                            upscale_folder_status = gr.Textbox(label="Status")
                            upscale_folder_gallery = gr.Gallery(label="Results", columns=3, height=240)
                    upscale_folder_evt = upscale_folder_button.click(
                        fn=_upscale_folder_stream,
                        inputs=[upscale_input_folder, upscale_folder_output, upscale_folder_upscaler, upscale_folder_scale, upscale_folder_quality],
                        outputs=[upscale_folder_status, upscale_folder_gallery],
                    )
                    upscale_folder_stop.click(fn=_stop_current_job, outputs=upscale_folder_status, cancels=[upscale_folder_evt])

        with gr.Tab("Video Upscale"):
            with gr.Row():
                with gr.Column():
                    video_upscale_input = gr.File(label="Video file", file_count="single", file_types=VIDEO_EXTS, type="filepath")
                    video_upscale_output, _ = directory_field("Output folder", str(DEFAULT_OUTPUT_DIR), state_key="video_upscale_output")
                    video_upscale_upscaler = gr.Dropdown(label="Upscaler", choices=UPSCALER_CHOICES, value="rtx")
                    video_upscale_scale = gr.Number(label="Scale", value=2.0, precision=2)
                    video_upscale_quality = gr.Dropdown(label="RTX quality", choices=QUALITY_CHOICES, value="ULTRA")
                    video_upscale_button = gr.Button("Upscale")
                    video_upscale_stop = gr.Button("Stop")
                    video_upscale_status = gr.Textbox(label="Status")
                with gr.Column():
                    video_upscale_result = gr.Video(label="Output video")
            video_upscale_evt = video_upscale_button.click(
                fn=_video_upscale_handler,
                inputs=[video_upscale_input, video_upscale_output, video_upscale_upscaler, video_upscale_scale, video_upscale_quality],
                outputs=[video_upscale_result, video_upscale_status],
            )
            video_upscale_stop.click(fn=_stop_current_job, outputs=video_upscale_status, cancels=[video_upscale_evt])

        with gr.Tab("Manage Models"):
            with gr.Row():
                with gr.Column():
                    model_refresh = gr.Button("Refresh")
                    model_delete = gr.Button("Delete selected")
                    model_cleanup_button = gr.Button("Remove unused / duplicate models")
                    model_cleanup_confirm = gr.Button("Confirm removal", visible=False)
                    model_total = gr.Markdown("**Total installed model files:** 0  \n**Disk usage:** 0 B")
                    model_cleanup_preview = gr.Markdown("**No cleanup preview yet.**")
                    model_status = gr.Textbox(label="Status")
                with gr.Column():
                    model_list = gr.CheckboxGroup(label="Installed models", choices=[], value=[])
                    model_cleanup_state = gr.State([])
            model_refresh.click(fn=_model_manager_refresh, outputs=[model_list, model_total, model_status])
            model_delete.click(fn=_model_manager_delete, inputs=[model_list], outputs=[model_list, model_total, model_status])
            model_cleanup_button.click(
                fn=_model_manager_cleanup_preview,
                outputs=[model_cleanup_preview, model_status, model_cleanup_confirm, model_cleanup_state],
            )
            model_cleanup_confirm.click(
                fn=_model_manager_cleanup_confirm,
                inputs=[model_cleanup_state],
                outputs=[model_list, model_total, model_cleanup_preview, model_status, model_cleanup_confirm, model_cleanup_state],
            )

        def bind_textbox(component, key: str) -> None:
            component.input(
                fn=lambda state, value: _ui_state_update(state, key, value or ""),
                inputs=[ui_state, component],
                outputs=[ui_state],
            )

        def bind_value(component, key: str) -> None:
            component.change(
                fn=lambda state, value: _ui_state_update(state, key, value),
                inputs=[ui_state, component],
                outputs=[ui_state],
            )

        def bind_path(component, key: str) -> None:
            def persist(state, value):
                if value in (None, ""):
                    return _ui_state_update(state, key, None)
                try:
                    path = _component_path(value)
                except Exception:
                    return _ui_state_update(state, key, None)
                return _ui_state_update(state, key, str(path) if path.exists() else None)

            component.change(fn=persist, inputs=[ui_state, component], outputs=[ui_state])

        def bind_shared_text(*components) -> None:
            outputs = [*components, ui_state]

            def sync(state, value):
                text = value or ""
                updated = _ui_state_update(state, "shared_prompt", text)
                return (*([text] * len(components)), updated)

            for component in components:
                component.input(fn=sync, inputs=[ui_state, component], outputs=outputs)

        def bind_shared_negative(*components) -> None:
            outputs = [*components, ui_state]

            def sync(state, value):
                text = value or ""
                updated = _ui_state_update(state, "shared_negative", text)
                return (*([text] * len(components)), updated)

            for component in components:
                component.input(fn=sync, inputs=[ui_state, component], outputs=outputs)

        bind_shared_text(edit_prompt, batch_prompt, t2i_prompt, video_prompt)
        bind_shared_negative(edit_negative, batch_negative, t2i_negative, video_negative)
        bind_path(edit_image, "edit_image")
        bind_path(edit_reference, "edit_reference")
        bind_textbox(frame_dir, "video_frame_dir")
        bind_path(video_input, "video_input")
        bind_path(upscale_image, "upscale_image")
        bind_path(video_upscale_input, "video_upscale_input")
        bind_value(every_n, "video_every_n")
        bind_value(max_frames, "video_max_frames")
        bind_value(video_steps, "video_steps")
        bind_value(video_cfg, "video_cfg")
        bind_value(video_megapixels, "video_megapixels")
        bind_value(video_seed, "video_seed")
        bind_value(video_engine, "video_engine")
        bind_value(video_compile, "video_compile")
        bind_value(video_mrflow, "video_mrflow")
        bind_value(edit_steps, "edit_steps")
        bind_value(edit_cfg, "edit_cfg")
        bind_value(edit_megapixels, "edit_megapixels")
        bind_value(edit_seed, "edit_seed")
        bind_value(edit_engine, "edit_engine")
        bind_value(edit_live_preview, "edit_live_preview")
        bind_value(edit_consistency, "edit_consistency")
        bind_value(edit_consistency_strength, "edit_consistency_strength")
        bind_value(edit_compile, "edit_compile")
        bind_value(edit_mrflow, "edit_mrflow")
        bind_value(edit_teacache, "edit_teacache")
        bind_value(edit_depth_lock, "edit_depth_lock")
        bind_value(batch_steps, "batch_steps")
        bind_value(batch_cfg, "batch_cfg")
        bind_value(batch_megapixels, "batch_megapixels")
        bind_value(batch_seed, "batch_seed")
        bind_value(batch_engine, "batch_engine")
        bind_value(batch_consistency, "batch_consistency")
        bind_value(batch_consistency_strength, "batch_consistency_strength")
        bind_value(batch_compile, "batch_compile")
        bind_value(batch_teacache, "batch_teacache")
        bind_value(batch_mrflow, "batch_mrflow")
        bind_value(t2i_width, "t2i_width")
        bind_value(t2i_height, "t2i_height")
        bind_value(t2i_steps, "t2i_steps")
        bind_value(t2i_cfg, "t2i_cfg")
        bind_value(t2i_seed, "t2i_seed")
        bind_value(t2i_engine, "t2i_engine")
        bind_value(t2i_live_preview, "t2i_live_preview")
        bind_value(t2i_consistency, "t2i_consistency")
        bind_value(t2i_consistency_strength, "t2i_consistency_strength")
        bind_value(t2i_compile, "t2i_compile")
        bind_value(t2i_mrflow, "t2i_mrflow")
        bind_value(t2i_teacache, "t2i_teacache")
        bind_value(upscale_upscaler, "upscale_upscaler")
        bind_value(upscale_scale, "upscale_scale")
        bind_value(upscale_quality, "upscale_quality")
        bind_value(upscale_folder_upscaler, "upscale_folder_upscaler")
        bind_value(upscale_folder_scale, "upscale_folder_scale")
        bind_value(upscale_folder_quality, "upscale_folder_quality")
        bind_value(video_upscale_upscaler, "video_upscale_upscaler")
        bind_value(video_upscale_scale, "video_upscale_scale")
        bind_value(video_upscale_quality, "video_upscale_quality")

        demo.load(
            fn=_restore_ui_state,
            inputs=[ui_state],
            outputs=[
                edit_prompt,
                batch_prompt,
                t2i_prompt,
                video_prompt,
                edit_negative,
                batch_negative,
                t2i_negative,
                video_negative,
                edit_image,
                edit_reference,
                edit_output,
                edit_steps,
                edit_cfg,
                edit_megapixels,
                edit_seed,
                edit_engine,
                edit_live_preview,
                edit_live_preview_image,
                edit_consistency,
                edit_consistency_strength,
                edit_compile,
                edit_mrflow,
                edit_teacache,
                edit_depth_lock,
                edit_depth_note,
                edit_depth_preview,
                video_input,
                video_output,
                video_steps,
                video_cfg,
                video_megapixels,
                video_seed,
                video_engine,
                video_compile,
                video_mrflow,
                every_n,
                max_frames,
                frame_dir,
                batch_input,
                batch_output,
                batch_steps,
                batch_cfg,
                batch_megapixels,
                batch_seed,
                batch_engine,
                batch_consistency,
                batch_consistency_strength,
                batch_compile,
                batch_teacache,
                batch_mrflow,
                t2i_output,
                t2i_width,
                t2i_height,
                t2i_steps,
                t2i_cfg,
                t2i_seed,
                t2i_engine,
                t2i_live_preview,
                t2i_consistency,
                t2i_consistency_strength,
                t2i_compile,
                t2i_mrflow,
                t2i_teacache,
                upscale_image,
                upscale_output,
                upscale_upscaler,
                upscale_scale,
                upscale_quality,
                upscale_input_folder,
                upscale_folder_output,
                upscale_folder_upscaler,
                upscale_folder_scale,
                upscale_folder_quality,
                video_upscale_input,
                video_upscale_output,
                video_upscale_upscaler,
                video_upscale_scale,
                video_upscale_quality,
                ui_state,
            ],
        )
        demo.load(fn=_model_manager_refresh, outputs=[model_list, model_total, model_status])
        demo.queue()
    return demo


if __name__ == "__main__":
    launch_host = os.environ.get("COMFYUI_UI_HOST", "127.0.0.1")
    launch_port = int(os.environ.get("COMFYUI_UI_PORT", "7861"))
    build_app().launch(server_name=launch_host, server_port=launch_port, share=False)
