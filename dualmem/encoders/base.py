"""VisualEncoder protocol shared by all encoder backends.

Design choice: encoders are RESPONSE-FREE — they don't fetch over a network
at construction time. Model weights are loaded lazily on the first
`embed_image` or `embed_text` call so unit tests that only need the
`dim` / `name` properties don't pay the download cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import Optional, Protocol, runtime_checkable

import numpy as np


class EncoderCapability(Flag):
    """What an encoder can do — used by retrievers to choose code paths."""
    IMAGE_ONLY = auto()
    TEXT_ONLY = auto()
    ALIGNED = auto()         # image + text in same vector space (CLIP-like)


@runtime_checkable
class VisualEncoder(Protocol):
    """Minimal contract any encoder must satisfy.

    All encoders return L2-normalized vectors so cosine similarity reduces
    to a dot product.
    """
    name: str
    dim: int
    capability: EncoderCapability

    def embed_image(self, image_path: str) -> np.ndarray: ...

    # Only encoders with EncoderCapability.ALIGNED or .TEXT_ONLY implement this.
    def embed_text(self, text: str) -> np.ndarray: ...


@dataclass
class EncoderConfig:
    """Common encoder-construction options."""
    device: str = "cpu"            # "cpu" / "cuda" — DINOv2/v3 will use whatever is available
    cache_dir: Optional[str] = None
    # If True, encoder may quietly fall back to stub when a model fails to load
    # (useful for CI machines without GPU and without HF_TOKEN).
    fallback_to_stub: bool = False


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
