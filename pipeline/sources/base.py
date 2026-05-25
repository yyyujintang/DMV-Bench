"""SourceProvider protocol — supplies the source image for edit-mode
runs. Implementations live in sibling modules:

  - file.py     : read existing PNG from a filesystem root
  - backend.py  : call another (t2i) backend on demand
  - template.py : (future) load a per-category template

A SourceProvider is opaque to the driver; the only contract is that
`source_for(spec)` returns raw image bytes (any common format; PIL
opens them later)."""
from __future__ import annotations

from typing import Any, Protocol


class SourceProvider(Protocol):
    provider_id: str

    def source_for(self, spec: dict[str, Any]) -> bytes:
        """Return raw image bytes for this variant spec.

        Should raise FileNotFoundError / RuntimeError on permanent
        misses. The driver does not retry source resolution.
        """
        ...
