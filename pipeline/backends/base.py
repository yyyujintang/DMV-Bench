"""Image-generation backend protocol.

The driver builds a GenerateRequest from the variant spec and calls
`backend.generate(req)`. The same call site handles both text-to-image
(no source image) and edit (source bytes provided). A backend declares
which modes it supports via `capabilities` and raises
BackendUnsupportedError if the driver misroutes a request.

Concrete implementations:

  - GeminiBackend       (text-to-image and edit — same REST endpoint)
  - QwenImageEditBackend (edit only — diffusers QwenImageEditPipeline)
  - future: FluxFill, SD3-Edit, etc.

A backend MUST NOT touch the filesystem, manifest, or QC. Its single
job is request-in → PNG-bytes-out. The driver owns IO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Literal, Protocol


Mode = Literal["text2image", "edit"]


class BackendError(RuntimeError):
    """Retryable backend failure (timeout, 5xx, no candidate)."""


class BackendUnsupportedError(BackendError):
    """The request used a mode this backend doesn't implement.
    Distinct from BackendError so the driver can fail fast instead of
    burning retries on a permanent mismatch."""


@dataclass(frozen=True)
class GenerateRequest:
    """The single argument every backend takes.

    For text-to-image, leave `source_image=None`. For edit, pass the
    raw source PNG bytes; `prompt` becomes the edit instruction.

    `negative_prompt` describes what the model should AVOID. It is
    only consulted by backends that support classifier-free guidance
    (Qwen-Image, Qwen-Image-Edit). Gemini ignores it. Without a
    negative_prompt the strict clauses inside `prompt` are weakly
    enforced; passing the same forbidden vocabulary here makes CFG
    actually push the latents away from them.
    """

    prompt: str
    seed: int | None = None
    source_image: bytes | None = None
    negative_prompt: str | None = None

    @property
    def mode(self) -> Mode:
        return "edit" if self.source_image is not None else "text2image"


class ImageBackend(Protocol):
    """Implementations attach the three attributes below as instance
    attributes so the driver can stamp them onto the manifest record."""

    backend_id: str               # e.g. "gemini-2.5-flash-image"
    model_revision: str
    capabilities: FrozenSet[Mode]  # subset of {"text2image", "edit"}

    def generate(self, req: GenerateRequest) -> bytes:
        """Return raw PNG bytes for the given request.

        Raise BackendUnsupportedError if `req.mode` isn't in this
        backend's capabilities. Raise BackendError on transient failure
        the driver can retry.
        """
        ...
