"""Disk-backed embedding cache.

The bottleneck of full-pipeline runs on CPU is re-encoding the same product
image dozens of times across (cells × systems). A simple `image_path → vec`
cache eliminates the redundancy and turns subsequent runs into pure VLM-API
time.

Cache schema (one .npy file per encoder + a JSON index):

    data/vismem_diag/_embed_cache/
        index.json                                   # {key: encoder_name}
        clip_clip-vit-base-patch32_<sha1>.npy        # one vector per image
        sbert_all-MiniLM-L6-v2_<sha1>.npy            # one vector per text

`sha1` is the SHA-1 of the input (image bytes or text string), so the same
image always lands in the same file regardless of full path.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np


class EmbedCache:
    """Process-wide disk cache for `image_path | text → np.ndarray` mappings.

    Use:
        cache = EmbedCache("data/vismem_diag/_embed_cache")
        vec = cache.get_or_compute(key, encoder_name,
                                    compute=lambda: encoder.embed_image(path))
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _key_for_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:16]

    @staticmethod
    def _key_for_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _path(self, encoder_name: str, key: str) -> Path:
        safe = encoder_name.replace(":", "_").replace("/", "_")
        return self.root / f"{safe}__{key}.npy"

    def get(self, key: str, encoder_name: str) -> Optional[np.ndarray]:
        p = self._path(encoder_name, key)
        if not p.exists(): return None
        try:
            return np.load(p)
        except Exception:
            return None

    def put(self, key: str, encoder_name: str, vec: np.ndarray) -> None:
        with self._lock:
            np.save(self._path(encoder_name, key), vec)

    def get_or_compute(self, key: str, encoder_name: str, compute) -> np.ndarray:
        v = self.get(key, encoder_name)
        if v is not None: return v
        v = compute()
        self.put(key, encoder_name, v)
        return v


# Singleton accessor — banks call this without configuration.
_default_cache: Optional[EmbedCache] = None


def get_default_cache() -> EmbedCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbedCache("data/vismem_diag/_embed_cache")
    return _default_cache
