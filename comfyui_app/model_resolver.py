from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from comfyui_app.config import COMFYUI_DIR, MODELS_DIR, REPO_ROOT, get_hf_token
from comfyui_app.vram import select_tier

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None  # type: ignore[assignment]


class ModelResolverError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Candidate:
    repo: str
    path_regex: str
    dest_subdir: str
    min_vram: float
    kind: str


MODEL_REGISTRY: dict[str, dict[str, list[Candidate]]] = {
    "diffusion": {
        "flux2_fp8": [
            Candidate(
                repo="black-forest-labs/FLUX.2-klein-4b-fp8",
                path_regex=r"(^|/)flux-2-klein-4b-fp8\.safetensors$",
                dest_subdir="diffusion_models",
                min_vram=7.0,
                kind="flux-2-klein-4b-fp8.safetensors",
            )
        ],
        "flux2_gguf_q4_k_m": [
            Candidate(
                repo="unsloth/FLUX.2-klein-4B-GGUF",
                path_regex=r"(^|/)flux-2-klein-4b-Q4_K_M\.gguf$",
                dest_subdir="diffusion_models",
                min_vram=5.0,
                kind="flux-2-klein-4b-Q4_K_M.gguf",
            )
        ],
        "flux2_gguf_q3_k_m": [
            Candidate(
                repo="unsloth/FLUX.2-klein-4B-GGUF",
                path_regex=r"(^|/)flux-2-klein-4b-Q3_K_M\.gguf$",
                dest_subdir="diffusion_models",
                min_vram=0.0,
                kind="flux-2-klein-4b-Q3_K_M.gguf",
            )
        ],
        "flux2_gguf_q2_k": [
            Candidate(
                repo="unsloth/FLUX.2-klein-4B-GGUF",
                path_regex=r"(^|/)flux-2-klein-4b-Q2_K\.gguf$",
                dest_subdir="diffusion_models",
                min_vram=0.0,
                kind="flux-2-klein-4b-Q2_K.gguf",
            )
        ],
        "nunchaku_int4": [
            Candidate(
                repo="tonera/FLUX.2-klein-4B-Nunchaku",
                path_regex=r"(^|/)svdq-int4_r32-FLUX\.2-klein-4B-Nunchaku\.safetensors$",
                dest_subdir="diffusion_models",
                min_vram=0.0,
                kind="svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors",
            )
        ],
    },
    "upscale": {
        "realesrgan_x2plus": [
            Candidate(
                repo="2kpr/Real-ESRGAN",
                path_regex=r"(^|/)RealESRGAN_x2plus\.pth$",
                dest_subdir="upscale_models",
                min_vram=0.0,
                kind="RealESRGAN_x2plus.pth",
            )
        ],
    },
    "text_encoder": {
        "flux2_fp4": [
            Candidate(
                repo="Comfy-Org/flux2-klein-4B",
                path_regex=r"(^|/)qwen_3_4b_fp4_flux2\.safetensors$",
                dest_subdir="text_encoders",
                min_vram=0.0,
                kind="qwen_3_4b_fp4_flux2.safetensors",
            )
        ],
        "flux2_full": [
            Candidate(
                repo="Comfy-Org/flux2-klein-4B",
                path_regex=r"(^|/)qwen_3_4b\.safetensors$",
                dest_subdir="text_encoders",
                min_vram=16.0,
                kind="qwen_3_4b.safetensors",
            )
        ],
    },
    "vae": {
        "flux2_small_decoder": [
            Candidate(
                repo="black-forest-labs/FLUX.2-small-decoder",
                path_regex=r"(^|/)full_encoder_small_decoder\.safetensors$",
                dest_subdir="vae",
                min_vram=0.0,
                kind="full_encoder_small_decoder.safetensors",
            )
        ],
        "flux2_vae": [
            Candidate(
                repo="Comfy-Org/flux2-klein-4B",
                path_regex=r"(^|/)flux2-vae\.safetensors$",
                dest_subdir="vae",
                min_vram=0.0,
                kind="flux2-vae.safetensors",
            )
        ],
    },
}

RESOLVED_MANIFEST = REPO_ROOT / "comfyui_app" / "resolved_models.json"
COMFYUI_MANIFEST = COMFYUI_DIR / "resolved_models.json"


def _fetch_repo_tree(repo: str, token: str | None) -> list[dict[str, object]]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ModelResolverError(
                f"Your Hugging Face token is missing, expired, or you haven't accepted the license for {repo}. "
                f"Visit https://huggingface.co/{repo} to accept terms, then re-run setup."
            ) from exc
        raise ModelResolverError(f"Could not read model files from Hugging Face for {repo}.") from exc
    except URLError as exc:
        raise ModelResolverError(f"Could not reach Hugging Face while checking files for {repo}.") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelResolverError(f"Hugging Face returned an unexpected response for {repo}.") from exc
    if not isinstance(data, list):
        raise ModelResolverError(f"Hugging Face returned an unexpected file list for {repo}.")
    return [entry for entry in data if isinstance(entry, dict)]


def _match_file(entries: Sequence[dict[str, object]], candidate: Candidate) -> dict[str, object]:
    pattern = re.compile(candidate.path_regex, re.IGNORECASE)
    matches = [entry for entry in entries if isinstance(entry.get("path"), str) and pattern.search(str(entry["path"]))]
    if not matches:
        raise ModelResolverError(
            f"Could not find {candidate.kind} in {candidate.repo}. Check the repository contents and try again."
        )

    def score(entry: dict[str, object]) -> tuple[int, int]:
        path = str(entry.get("path", ""))
        basename = Path(path).name
        exact_match = 1 if basename.lower() == candidate.kind.lower() else 0
        size_value = int(entry.get("size") or 0)
        return exact_match, size_value

    matches.sort(key=score, reverse=True)
    return matches[0]


def _candidate_to_resolved(candidate: Candidate, entry: Mapping[str, object]) -> dict[str, object]:
    remote_path = str(entry.get("path", ""))
    resolved = {
        "repo": candidate.repo,
        "remote_path": remote_path,
        "local_filename": Path(remote_path).name,
        "dest_dir": str(MODELS_DIR / candidate.dest_subdir),
        "size": int(entry.get("size") or 0),
        "sha": str(entry.get("oid") or entry.get("sha") or ""),
        "etag": str(entry.get("etag") or ""),
        "kind": candidate.kind,
    }
    return resolved


def _select_candidate_for_key(component: str, key: str, token: str | None) -> dict[str, object]:
    candidate_groups = MODEL_REGISTRY[component]
    if key not in candidate_groups:
        raise ModelResolverError(f"No model candidate is configured for {component}:{key}.")

    entries = _fetch_repo_tree(candidate_groups[key][0].repo, token)
    for candidate in candidate_groups[key]:
        entry = _match_file(entries, candidate)
        return _candidate_to_resolved(candidate, entry)
    raise ModelResolverError(f"Could not resolve a file for {component}:{key}.")


def resolve_models(
    vram_gb: float,
    token: str | None,
    prefer_gguf: bool = False,
    engine: str = "default",
) -> dict[str, dict[str, object]]:
    if not token:
        raise ModelResolverError(
            "A Hugging Face token is required to download the ComfyUI models. Run setup and paste your token first."
        )

    tier = select_tier(vram_gb)
    diffusion_key = "nunchaku_int4" if engine == "nunchaku_int4" else tier.diffusion
    if prefer_gguf and diffusion_key == "flux2_fp8":
        diffusion_key = "flux2_gguf_q4_k_m"
    text_encoder_key = tier.text_encoder

    resolved = {
        "diffusion": _select_candidate_for_key("diffusion", diffusion_key, token),
        "text_encoder": _select_candidate_for_key("text_encoder", text_encoder_key, token),
        "upscale": _select_candidate_for_key("upscale", "realesrgan_x2plus", token),
    }
    try:
        resolved["vae"] = _select_candidate_for_key("vae", "flux2_small_decoder", token)
    except ModelResolverError:
        resolved["vae"] = _select_candidate_for_key("vae", "flux2_vae", token)
    return resolved


def load_resolved_manifest(manifest_path: Path | None = None) -> dict[str, object] | None:
    paths = [manifest_path] if manifest_path is not None else [RESOLVED_MANIFEST, COMFYUI_MANIFEST]
    for path in paths:
        if path is None or not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _write_manifest(data: dict[str, object]) -> None:
    RESOLVED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    COMFYUI_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    RESOLVED_MANIFEST.write_text(payload, encoding="utf-8")
    COMFYUI_MANIFEST.write_text(payload, encoding="utf-8")


def download_models(
    resolved: Mapping[str, Mapping[str, object]],
    token: str | None,
    progress_cb: Callable[[str], None] | None = None,
    *,
    engine: str = "default",
) -> dict[str, dict[str, object]]:
    if hf_hub_download is None:
        raise ModelResolverError("huggingface_hub is not installed, so model downloads cannot continue.")

    manifest = {
        "engine": engine,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": {},
    }
    downloaded: dict[str, dict[str, object]] = {}
    for name, info in resolved.items():
        dest_dir = Path(str(info["dest_dir"]))
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote_path = str(info["remote_path"])
        local_filename = str(info["local_filename"])
        final_path = dest_dir / local_filename
        size = int(info.get("size") or 0)
        existing_manifest = load_resolved_manifest()
        previous = None
        if isinstance(existing_manifest, dict):
            models = existing_manifest.get("models")
            if isinstance(models, dict):
                previous = models.get(name)

        if (
            final_path.exists()
            and final_path.is_file()
            and final_path.stat().st_size == size
            and isinstance(previous, dict)
            and previous.get("repo") == info.get("repo")
            and previous.get("remote_path") == remote_path
            and int(previous.get("size") or 0) == size
        ):
            downloaded[name] = dict(info)
            manifest["models"][name] = dict(info)
            continue

        if progress_cb is not None:
            progress_cb(f"Downloading {name}: {local_filename}")
        downloaded_path = Path(
            hf_hub_download(
                repo_id=str(info["repo"]),
                filename=remote_path,
                token=token,
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False,
            )
        )
        if downloaded_path.name != local_filename:
            if final_path.exists():
                final_path.unlink()
            downloaded_path.replace(final_path)
        elif downloaded_path != final_path:
            if final_path.exists():
                final_path.unlink()
            downloaded_path.replace(final_path)
        info_copy = dict(info)
        info_copy["local_path"] = str(final_path)
        info_copy["downloaded"] = True
        downloaded[name] = info_copy
        manifest["models"][name] = info_copy
        if progress_cb is not None:
            progress_cb(f"Downloaded {name}")

    _write_manifest(manifest)
    return downloaded
