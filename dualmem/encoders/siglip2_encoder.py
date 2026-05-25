"""SigLIP-2 encoder — aligned image + text embeddings.

`google/siglip2-base-patch16-384` from HuggingFace (~600 MB, ~80M params,
aligned image+text encoder). The motivation for adding this alongside
`ClipEncoder` is to factor out the visual-encoder confound in the
DualChannel-vs-M2A-lite comparison: M2A-lite uses siglip2 internally,
so `DualChannel-siglip` keeps DualChannel's bank/retriever/injector
identical but swaps CLIP → siglip2 in the visual channel. The Δ vs
`DualChannel-norm` (which keeps CLIP) then isolates the encoder effect.

Lazy load: models materialize on first `embed_image` / `embed_text`,
GPU when CUDA is available, CPU fallback. Mirrors `ClipEncoder` in style.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PIL import Image

from dualmem.encoders.base import EncoderCapability, EncoderConfig, l2_normalize


class Siglip2Encoder:
    """SigLIP-2 via HuggingFace transformers."""
    name: str
    dim: int = 768
    capability = (
        EncoderCapability.ALIGNED | EncoderCapability.IMAGE_ONLY | EncoderCapability.TEXT_ONLY
    )

    def __init__(self, model: str = "google/siglip2-base-patch16-384",
                 config: Optional[EncoderConfig] = None):
        self.model_name = model
        self.name = f"siglip2:{model.split('/')[-1]}"
        self.config = config or EncoderConfig()
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModel, AutoProcessor
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._model.eval()
            try:
                self._model.to(self.config.device)
            except Exception:
                self.config.device = "cpu"
            try:
                self.dim = int(self._model.config.text_config.hidden_size)
            except Exception:
                self.dim = 768

    def embed_image(self, image_path: str) -> np.ndarray:
        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return l2_normalize(feats[0].cpu().numpy().astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(text=[text], padding="max_length",
                                  return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return l2_normalize(feats[0].cpu().numpy().astype(np.float32))
