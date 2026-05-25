"""Existing-file source provider.

Reads `<root>/<imagePath>` from disk. Useful when iterating on an
image-edit backend over the current live image tree (e.g. taking the
Gemini Phase-A outputs and edit-refining them with Qwen-Image-Edit
without re-paying Gemini API costs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ExistingFileSourceProvider:
    provider_id = "existing-file"

    def __init__(self, root: Path):
        self.root = Path(root)

    def source_for(self, spec: dict[str, Any]) -> bytes:
        rel = spec["imagePath"].lstrip("/")
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"no source image at {path} (provider=existing-file)"
            )
        return path.read_bytes()
