"""Backend registry. New backends drop in by adding a lazy branch to
`make_backend()` and a concrete module under this package.

Imports are deferred per-branch so a Gemini-only run never pulls in
diffusers/torch (cost: many seconds + many GB resident memory)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BackendError,
    BackendUnsupportedError,
    GenerateRequest,
    ImageBackend,
    Mode,
)

if TYPE_CHECKING:
    # Only for static type checkers; never executed at runtime.
    from .gemini import GeminiBackend  # noqa: F401
    from .qwen_image import QwenImageBackend  # noqa: F401
    from .qwen_image_edit import QwenImageEditBackend  # noqa: F401


def make_backend(name: str, **kwargs) -> ImageBackend:
    """Construct a backend by name. Unknown names raise ValueError.

    Known names:
      - "gemini"            → GeminiBackend (text2image, edit)
      - "qwen-image"        → QwenImageBackend (text2image only, diffusers)
      - "qwen-image-edit"   → QwenImageEditBackend (edit only, defaults to
                              the Plus / 2509 variant)
    """
    if name == "gemini":
        from .gemini import GeminiBackend
        return GeminiBackend(**kwargs)
    if name == "qwen-image":
        from .qwen_image import QwenImageBackend
        return QwenImageBackend(**kwargs)
    if name == "qwen-image-edit":
        from .qwen_image_edit import QwenImageEditBackend
        return QwenImageEditBackend(**kwargs)
    raise ValueError(
        f"unknown backend: {name!r}. "
        f"known: 'gemini', 'qwen-image', 'qwen-image-edit'"
    )


__all__ = [
    "BackendError",
    "BackendUnsupportedError",
    "GenerateRequest",
    "ImageBackend",
    "Mode",
    "make_backend",
]
