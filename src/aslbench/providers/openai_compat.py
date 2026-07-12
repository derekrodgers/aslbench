"""OpenAI-compatible provider.

Covers OpenAI, LM Studio, and any server exposing an OpenAI-style endpoint.
The only differences between them are ``base_url`` and the API key.

Images are sent as inline base64 data URLs, so the filename (which encodes the
class in this benchmark) is never transmitted.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from ..config import IMAGE_MEDIA_TYPE
from ..config import ProviderConfig
from . import CompletionResult, ModelInfo, retry_call

MAX_TOKENS = 2000


def _encode_image(image_path: Path) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig):
        self.id = cfg.id
        self.label = cfg.label
        self._cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = self._cfg.resolve_api_key() or "not-needed"
            self._client = OpenAI(base_url=self._cfg.base_url, api_key=api_key)
        return self._client

    def is_configured(self) -> bool:
        return self._cfg.credential_present()

    def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        resp = client.models.list()
        models = []
        for m in resp.data:
            models.append(ModelInfo(id=m.id, label=m.id, vision=None))
        return models

    def complete(self, model: str, prompt: str, image_path: Path) -> CompletionResult:
        client = self._get_client()
        b64 = _encode_image(image_path)
        data_url = f"data:{IMAGE_MEDIA_TYPE};base64,{b64}"

        def _call() -> CompletionResult:
            start = time.monotonic()
            kwargs = dict(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
            )
            try:
                resp = client.chat.completions.create(temperature=0, **kwargs)
            except Exception:
                # Some endpoints reject temperature; retry without it.
                resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - start
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            return CompletionResult(
                text=text,
                latency_s=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        return retry_call(_call)
