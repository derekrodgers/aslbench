"""Provider protocol, data types, registry, and factory.

A provider abstracts one API surface (OpenAI-compatible, Anthropic, Copilot
SDK). Models are never declared statically; the app enumerates them live via
``list_models``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from ..config import ProviderConfig

# Sentinel prefix marking a CompletionResult that failed after all retries.
ERROR_SENTINEL = "__PROVIDER_ERROR__"


@dataclass
class ModelInfo:
    id: str
    label: str
    vision: bool | None  # None = capability unknown


@dataclass
class CompletionResult:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None  # non-None when the call failed after retries
    thinking: str | None = None  # reasoning trace, when provided separately


@runtime_checkable
class Provider(Protocol):
    id: str
    label: str

    def is_configured(self) -> bool: ...

    def list_models(self) -> list[ModelInfo]: ...

    def complete(self, model: str, prompt: str, image_path: Path) -> CompletionResult: ...


def retry_call(
    fn: Callable[[], CompletionResult],
    retries: int = 2,
    backoff: tuple[float, ...] = (2.0, 8.0),
) -> CompletionResult:
    """Call ``fn`` with up to ``retries`` retries on exception.

    On final failure returns a CompletionResult carrying an error sentinel so
    the runner records a failed item rather than crashing.
    """
    last_exc: Exception | None = None
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider transport errors vary
            last_exc = exc
            if attempt < retries:
                delay = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                time.sleep(delay)
    return CompletionResult(
        text=f"{ERROR_SENTINEL}: {last_exc}",
        latency_s=0.0,
        error=str(last_exc),
    )


def build_provider(cfg: ProviderConfig) -> Provider:
    """Instantiate the provider implementation for a config entry."""
    if cfg.type == "openai_compatible":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    if cfg.type == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.type == "copilot_sdk":
        from .copilot_provider import CopilotProvider

        return CopilotProvider(cfg)
    raise ValueError(f"Unknown provider type: {cfg.type}")


# Cache providers by id so the app reuses one instance (and thus one Copilot
# event loop) across callbacks.
_PROVIDER_CACHE: dict[str, Provider] = {}


def get_provider(cfg: ProviderConfig) -> Provider:
    if cfg.id not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[cfg.id] = build_provider(cfg)
    return _PROVIDER_CACHE[cfg.id]
