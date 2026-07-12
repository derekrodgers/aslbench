"""GitHub Copilot SDK provider.

The Copilot Python SDK is async. This provider owns a dedicated asyncio event
loop running on a background thread and exposes the synchronous ``Provider``
interface by scheduling coroutines onto that loop. One ``CopilotClient`` is
started lazily on first use and reused across calls.

Each ``complete`` call creates a per-item session with an empty tool allowlist
(``available_tools=[]``) so the CLI's agentic tool loop is suppressed and each
item is a single-shot vision completion; see copilot-sdk-docs/features/agent-loop.md
and copilot-sdk-docs/features/image-input.md.

The SDK attaches images by file path, and the dataset filenames encode the
class (for example ``P1_A_5.jpg``). To avoid leaking the answer, each image is
first copied to a temporary file with a neutral name before it is attached; the
copy is deleted afterwards.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from ..config import IMAGE_SUFFIX, ProviderConfig
from . import CompletionResult, ModelInfo, retry_call

# Per-item completion timeout, seconds.
COMPLETE_TIMEOUT_S = 180.0
# Client start / auth probe timeout, seconds.
START_TIMEOUT_S = 30.0


class _Loop:
    """A background thread hosting a single asyncio event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)


def _extract_vision(model) -> bool | None:
    """Best-effort read of capabilities.supports.vision from a model object."""
    caps = getattr(model, "capabilities", None)
    if caps is None and isinstance(model, dict):
        caps = model.get("capabilities")
    if caps is None:
        return None
    supports = getattr(caps, "supports", None)
    if supports is None and isinstance(caps, dict):
        supports = caps.get("supports")
    if supports is None:
        return None
    vision = getattr(supports, "vision", None)
    if vision is None and isinstance(supports, dict):
        vision = supports.get("vision")
    return bool(vision) if vision is not None else None


def _model_id(model) -> str:
    return getattr(model, "id", None) or (model.get("id") if isinstance(model, dict) else str(model))


def _model_label(model) -> str:
    label = getattr(model, "name", None) or getattr(model, "display_name", None)
    if label is None and isinstance(model, dict):
        label = model.get("name") or model.get("display_name")
    return label or _model_id(model)


class CopilotProvider:
    def __init__(self, cfg: ProviderConfig):
        self.id = cfg.id
        self.label = cfg.label
        self._cfg = cfg
        self._loop: _Loop | None = None
        self._client = None
        self._client_lock = threading.Lock()
        self._configured: bool | None = None

    def _get_loop(self) -> _Loop:
        if self._loop is None:
            self._loop = _Loop()
        return self._loop

    def _ensure_client(self):
        """Start the CopilotClient once, on the background loop."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            from copilot import CopilotClient

            loop = self._get_loop()

            async def _start():
                client = CopilotClient()
                await client.start()
                return client

            self._client = loop.run(_start(), timeout=START_TIMEOUT_S)
        return self._client

    def is_configured(self) -> bool:
        if self._configured is not None:
            return self._configured
        try:
            self._ensure_client()
            self._configured = True
        except Exception:
            self._configured = False
        return self._configured

    def list_models(self) -> list[ModelInfo]:
        client = self._ensure_client()
        loop = self._get_loop()

        async def _list():
            return await client.list_models()

        raw = loop.run(_list(), timeout=START_TIMEOUT_S)
        models = []
        for m in raw:
            vision = _extract_vision(m)
            models.append(ModelInfo(id=_model_id(m), label=_model_label(m), vision=vision))
        return models

    def complete(self, model: str, prompt: str, image_path: Path) -> CompletionResult:
        client = self._ensure_client()
        loop = self._get_loop()

        # Copy the image to a neutral filename so the class-encoding original
        # name is never revealed to the model through the attachment path.
        tmp_dir = Path(tempfile.mkdtemp(prefix="aslbench-"))
        neutral = tmp_dir / f"query-{uuid.uuid4().hex}{IMAGE_SUFFIX}"
        shutil.copy2(image_path, neutral)
        abs_path = str(neutral.resolve())

        def _call() -> CompletionResult:
            async def _do() -> CompletionResult:
                from copilot.session import PermissionHandler

                collected: list[str] = []

                session = await client.create_session(
                    on_permission_request=PermissionHandler.approve_all,
                    model=model,
                    available_tools=[],
                )
                try:
                    response = await session.send_and_wait(
                        prompt,
                        attachments=[
                            {"type": "file", "path": abs_path, "displayName": "image.jpg"}
                        ],
                    )
                    text = ""
                    if response is not None:
                        data = getattr(response, "data", None)
                        if data is not None:
                            text = getattr(data, "content", "") or ""
                    if not text and collected:
                        text = "".join(collected)
                    return CompletionResult(text=text, latency_s=0.0)
                finally:
                    destroy = getattr(session, "destroy", None) or getattr(session, "disconnect", None)
                    if destroy is not None:
                        try:
                            await destroy()
                        except Exception:
                            pass

            start = time.monotonic()
            result = loop.run(_do(), timeout=COMPLETE_TIMEOUT_S)
            result.latency_s = time.monotonic() - start
            return result

        try:
            return retry_call(_call)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
