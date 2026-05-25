"""Fixed-file source provider.

Returns the SAME image bytes for every spec — regardless of urlHash
or categorySlug. Use this for the "anchor + variant" pattern: a
single Qwen-Image t2i output becomes the seed for many Qwen-Image-Edit
calls that re-style it without re-rolling the underlying composition.

This is different from ExistingFileSourceProvider, which reads the
spec's own `imagePath`. For the anchor pattern we want every variant
in a category to share ONE canonical source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class FixedFileSourceProvider:
    provider_id = "fixed-file"

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"FixedFileSourceProvider: source file not found: {self.path}"
            )
        # Cache bytes — we'll be called many times with the same content.
        self._bytes: bytes = self.path.read_bytes()

    def source_for(self, spec: dict[str, Any]) -> bytes:  # noqa: ARG002
        return self._bytes
