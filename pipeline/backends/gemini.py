"""Gemini image-generation backend.

`gemini-2.5-flash-image` supports both text-to-image and instruction-
based image editing through the same `:generateContent` REST endpoint:

  - text2image: contents.parts = [{ text: <prompt> }]
  - edit:       contents.parts = [{ text: <instruction> },
                                  { inlineData: { mimeType, data } }]

We accept a GenerateRequest and pick the right body shape from
`req.mode`. The backend has no retry logic of its own; transient
failures (timeout, network, HTTP 5xx, NO_IMAGE) surface as
BackendError so the driver's retry loop owns recovery.

Note: gemini-2.5-flash-image does NOT honour seeds. The seed is
recorded on the manifest so a future seed-honouring backend can
replay the same logical generation."""
from __future__ import annotations

import base64
import json
import os
import socket
from urllib import error, request

from .base import (
    BackendError,
    BackendUnsupportedError,
    GenerateRequest,
    ImageBackend,
)


DEFAULT_MODEL = "gemini-2.5-flash-image"
ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiBackend:
    """ImageBackend backed by the Gemini REST API. Supports both modes."""

    capabilities = frozenset({"text2image", "edit"})

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise BackendError(
                "GEMINI_API_KEY not set — export it before instantiating "
                "GeminiBackend (typically via the repo-root .env)."
            )
        self.model = model or os.environ.get("GEMINI_IMAGE_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.backend_id = self.model
        self.model_revision = (
            f"env:{os.environ['GEMINI_IMAGE_MODEL']}"
            if "GEMINI_IMAGE_MODEL" in os.environ
            else "default"
        )

    def generate(self, req: GenerateRequest) -> bytes:
        if req.mode not in self.capabilities:
            raise BackendUnsupportedError(
                f"GeminiBackend does not support mode {req.mode!r}"
            )
        parts: list[dict] = [{"text": req.prompt}]
        if req.mode == "edit":
            # Gemini expects base64-encoded PNG bytes alongside the
            # instruction. We assume PNG; the existing pipeline only
            # produces PNG.
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(req.source_image).decode("ascii"),
                },
            })
        body = json.dumps({
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }).encode("utf-8")
        http_req = request.Request(
            ENDPOINT_TEMPLATE.format(model=self.model),
            data=body,
            headers={
                "x-goog-api-key": self.api_key,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:500]
            raise BackendError(f"HTTP {e.code}: {body_text}") from e
        except (TimeoutError, socket.timeout) as e:
            raise BackendError(f"socket timeout after {self.timeout}s") from e
        except (error.URLError, ConnectionError) as e:
            raise BackendError(f"network error: {e}") from e
        for cand in payload.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        raise BackendError(
            "No inlineData in response: " + json.dumps(payload)[:600]
        )
