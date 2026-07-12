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

MAX_TOKENS_DEFAULT = 16384  # generous default; thinking models need headroom


def _encode_image(image_path: Path) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig):
        self.id = cfg.id
        self.label = cfg.label
        self._cfg = cfg
        self._client = None
        # Configurable via  extra.max_tokens  in providers.yaml.
        # Thinking models (Qwen3, DeepSeek-R1, …) stream the full reasoning
        # trace into message.content and can easily exceed 2 000 tokens before
        # reaching the ANSWER line, so the default is intentionally high.
        self._max_tokens: int = int(cfg.extra.get("max_tokens", MAX_TOKENS_DEFAULT))
        # Arbitrary extra fields merged into every chat-completions request body.
        # Useful for server-specific knobs such as thinking budget (see providers.yaml).
        self._extra_body: dict = cfg.extra.get("extra_body", {})

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
        max_tokens = self._max_tokens

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
                max_tokens=max_tokens,
            )
            if self._extra_body:
                kwargs["extra_body"] = self._extra_body
            try:
                resp = client.chat.completions.create(temperature=0, **kwargs)
            except Exception:
                # Some endpoints reject temperature; retry without it.
                resp = client.chat.completions.create(**kwargs)
            latency = time.monotonic() - start
            text = resp.choices[0].message.content or ""
            finish_reason = resp.choices[0].finish_reason

            # Capture the reasoning/thinking trace when the server separates it
            # from the final answer (DeepSeek-style field used by oMLX, vLLM, …).
            msg = resp.choices[0].message
            thinking: str = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "thinking", None)
                or ""
            )
            thinking = thinking.strip()

            # Fallback: some servers embed thinking inside <think>…</think> tags
            # at the start of message.content.  Extract and remove those tags.
            if not thinking and text.startswith("<think>"):
                import re as _re
                m = _re.match(r"<think>(.*?)</think>\s*", text, _re.DOTALL | _re.IGNORECASE)
                if m:
                    thinking = m.group(1).strip()
                    text = text[m.end():].strip()

            if finish_reason == "length":
                # Generation stopped at the token cap before the model could
                # write ANSWER: X.  Tag the response so it shows up visibly in
                # the item explorer rather than looking like a plain parse-failure.
                text += (
                    f"\n\n[TRUNCATED — hit max_tokens={max_tokens}. "
                    "Increase via  extra.max_tokens  in providers.yaml.]"
                )
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            return CompletionResult(
                text=text,
                latency_s=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
                thinking=thinking or None,
            )

        return retry_call(_call)
