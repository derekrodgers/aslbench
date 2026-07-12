"""Anthropic provider.

Images are sent as inline base64 blocks, so the filename (which encodes the
class in this benchmark) is never transmitted.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from ..config import IMAGE_MEDIA_TYPE, ProviderConfig
from . import CompletionResult, ModelInfo, retry_call

MAX_TOKENS = 2000


def _encode_image(image_path: Path) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig):
        self.id = cfg.id
        self.label = cfg.label
        self._cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._cfg.resolve_api_key())
        return self._client

    def is_configured(self) -> bool:
        return self._cfg.credential_present()

    def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        resp = client.models.list()
        models = []
        for m in resp.data:
            label = getattr(m, "display_name", None) or m.id
            # All current Claude models accept images.
            models.append(ModelInfo(id=m.id, label=label, vision=True))
        return models

    def complete(self, model: str, prompt: str, image_path: Path) -> CompletionResult:
        client = self._get_client()
        b64 = _encode_image(image_path)

        def _call() -> CompletionResult:
            start = time.monotonic()
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": IMAGE_MEDIA_TYPE,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            latency = time.monotonic() - start
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage else None
            out_tok = getattr(usage, "output_tokens", None) if usage else None
            return CompletionResult(
                text=text,
                latency_s=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        return retry_call(_call)
