from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
import struct
import time
import uuid
from pathlib import Path
from typing import Callable

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

try:
    from websocket import create_connection
except Exception:  # pragma: no cover - optional dependency
    create_connection = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PromptImage:
    filename: str
    subfolder: str
    image_type: str


class ComfyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = uuid.uuid4().hex
        self._session = requests.Session() if requests is not None else None

    def _require_requests(self) -> None:
        if requests is None or self._session is None:
            raise RuntimeError("The requests package is required for ComfyUI communication.")

    def is_server_up(self) -> bool:
        self._require_requests()
        try:
            response = self._session.get(f"{self.base_url}/system_stats", timeout=3)
            return response.ok
        except Exception:
            return False

    def wait_until_up(self, timeout: float = 120.0, poll_interval: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_server_up():
                return
            time.sleep(poll_interval)
        raise RuntimeError(f"ComfyUI did not respond at {self.base_url} within {timeout:.0f} seconds.")

    def upload_image(self, path: Path | str) -> str:
        self._require_requests()
        image_path = Path(path)
        with image_path.open("rb") as handle:
            files = {"image": (image_path.name, handle, "application/octet-stream")}
            data = {"overwrite": "true", "subfolder": ""}
            response = self._session.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=120)
        response.raise_for_status()
        payload = self._parse_json(response.text)
        if isinstance(payload, dict):
            for key in ("name", "filename", "path"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
        raise RuntimeError("ComfyUI did not return an uploaded image name.")

    def queue_prompt(self, prompt_dict: dict[str, object], client_id: str | None = None) -> str:
        self._require_requests()
        payload = {"prompt": prompt_dict, "client_id": client_id or self.client_id}
        response = self._session.post(f"{self.base_url}/prompt", json=payload, timeout=120)
        response.raise_for_status()
        data = self._parse_json(response.text)
        if isinstance(data, dict):
            for key in ("prompt_id", "id"):
                prompt_id = data.get(key)
                if isinstance(prompt_id, str) and prompt_id:
                    return prompt_id
        raise RuntimeError("ComfyUI did not return a prompt id.")

    def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str | None = None,
        timeout: float = 600.0,
        preview_callback: Callable[[object], None] | None = None,
    ) -> None:
        if create_connection is not None:
            try:
                self._wait_with_websocket(prompt_id, client_id or self.client_id, timeout, preview_callback=preview_callback)
                return
            except Exception as exc:
                logger.debug("Websocket wait failed; falling back to polling: %s", exc)
        self._wait_with_polling(prompt_id, timeout)

    def _wait_with_websocket(
        self,
        prompt_id: str,
        client_id: str,
        timeout: float,
        preview_callback: Callable[[object], None] | None = None,
    ) -> None:
        if create_connection is None:
            raise RuntimeError("websocket-client is not installed.")
        ws = create_connection(f"{self.ws_url}?clientId={client_id}", timeout=timeout)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                message = ws.recv()
                if not message:
                    continue
                if isinstance(message, (bytes, bytearray, memoryview)):
                    preview_image = self._decode_preview_bytes(bytes(message))
                    if preview_image is not None and preview_callback is not None:
                        try:
                            preview_callback(preview_image)
                        except Exception:
                            logger.debug("Preview callback failed", exc_info=True)
                    continue
                data = self._parse_json(message)
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "executing":
                    payload = data.get("data")
                    if isinstance(payload, dict) and payload.get("prompt_id") == prompt_id and payload.get("node") is None:
                        return
            raise RuntimeError(f"ComfyUI did not finish prompt {prompt_id} within the timeout.")
        finally:
            ws.close()

    def _decode_preview_bytes(self, payload: bytes) -> object | None:
        if Image is None or len(payload) <= 8:
            return None
        try:
            event_type = struct.unpack(">I", payload[:4])[0]
            image_type = struct.unpack(">I", payload[4:8])[0]
        except struct.error:
            return None
        if event_type not in (1, 2):
            return None
        if image_type not in (1, 2):
            return None
        try:
            return Image.open(io.BytesIO(payload[8:])).convert("RGB")
        except Exception:
            logger.debug("Failed to decode ComfyUI preview frame", exc_info=True)
            return None

    def _wait_with_polling(self, prompt_id: str, timeout: float) -> None:
        self._require_requests()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self._session.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            if response.ok:
                data = self._parse_json(response.text)
                if isinstance(data, dict) and prompt_id in data:
                    prompt_entry = data.get(prompt_id)
                    if isinstance(prompt_entry, dict):
                        outputs = prompt_entry.get("outputs")
                        if isinstance(outputs, dict):
                            return
            time.sleep(2.0)
        raise RuntimeError(f"ComfyUI did not finish prompt {prompt_id} within the timeout.")

    def get_images(self, prompt_id: str) -> list[object]:
        self._require_requests()
        response = self._session.get(f"{self.base_url}/history/{prompt_id}", timeout=120)
        response.raise_for_status()
        data = self._parse_json(response.text)
        images: list[object] = []
        if not isinstance(data, dict):
            return images
        prompt_entry = data.get(prompt_id)
        if not isinstance(prompt_entry, dict):
            return images
        outputs = prompt_entry.get("outputs")
        if not isinstance(outputs, dict):
            return images
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            for item in output.get("images", []):
                if not isinstance(item, dict):
                    continue
                image = self._fetch_image(item)
                if image is not None:
                    images.append(image)
        return images

    def interrupt(self) -> None:
        self._require_requests()
        try:
            self._session.post(f"{self.base_url}/interrupt", timeout=10)
        except Exception:
            logger.debug("ComfyUI interrupt request failed", exc_info=True)

    def _fetch_image(self, item: dict[str, object]) -> object | None:
        filename = item.get("filename")
        subfolder = item.get("subfolder", "")
        image_type = item.get("type", "output")
        if not isinstance(filename, str):
            return None
        if Image is None:
            raise RuntimeError("The Pillow package is required to read images from ComfyUI.")
        params = {"filename": filename, "subfolder": str(subfolder), "type": str(image_type)}
        response = self._session.get(f"{self.base_url}/view", params=params, timeout=120)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    @staticmethod
    def _parse_json(payload: str) -> object:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
