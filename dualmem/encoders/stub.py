"""Deterministic hash-based encoder for CI / unit tests.

The same image always produces the same vector. Vectors for two different
images are loosely orthogonal but not perceptually meaningful, so the
stub is NOT a substitute for a real encoder during evaluation — it is
only for plumbing tests.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dualmem.encoders.base import EncoderCapability, VisualEncoder, l2_normalize


def _hash_to_vec(seed: bytes, dim: int) -> np.ndarray:
    """SHA-256-derived float vector, deterministic."""
    out = np.zeros(dim, dtype=np.float32)
    # Repeat-hash to fill `dim` floats.
    h = hashlib.sha256(seed).digest()
    raw = b""
    counter = 0
    while len(raw) < dim * 4:
        raw += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
        counter += 1
    out[:] = np.frombuffer(raw[: dim * 4], dtype=np.uint32).astype(np.float32) / 2**32 - 0.5
    return l2_normalize(out)


class StubEncoder:
    """Hash-based encoder. `supports_text_query == True` (text→image bridge is
    a no-op hash so the retriever can exercise the aligned-encoder path)."""
    name = "stub-v0"
    dim = 64
    capability = EncoderCapability.ALIGNED | EncoderCapability.IMAGE_ONLY | EncoderCapability.TEXT_ONLY

    def embed_image(self, image_path: str) -> np.ndarray:
        with open(image_path, "rb") as f:
            data = f.read()
        return _hash_to_vec(data, self.dim)

    def embed_text(self, text: str) -> np.ndarray:
        return _hash_to_vec(text.encode("utf-8"), self.dim)
