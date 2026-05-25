"""SBERT (sentence-transformers) — text-only encoder.

Used for verbal-channel retrieval whenever the memory bank stores
captions or fact strings. Default model is `all-MiniLM-L6-v2` (384d,
~22M params, very fast on CPU).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from dualmem.encoders.base import EncoderCapability, EncoderConfig, l2_normalize


class SbertEncoder:
    """SentenceTransformers wrapper."""
    name: str
    dim: int = 384
    capability = EncoderCapability.TEXT_ONLY

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 config: Optional[EncoderConfig] = None):
        self.model_name = model
        self.name = f"sbert:{model.split('/')[-1]}"
        self.config = config or EncoderConfig()
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, cache_folder=self.config.cache_dir)
            try:
                self._model = self._model.to(self.config.device)
            except Exception:
                pass
            self.dim = self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        v = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return v.astype(np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError("SBERT is text-only. Use a CLIP/DINOv2/v3 encoder for images.")
