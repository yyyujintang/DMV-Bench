"""CLIP encoder — aligned image + text embeddings.

We use HuggingFace `transformers` (already installed) so we avoid the
`open_clip` dependency. Default model is OpenAI's `clip-vit-base-patch32`
(512d, ~150M params, runs comfortably on CPU at ~50-100ms/image).

The encoder is lazy: models load on first `embed_image` / `embed_text`.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PIL import Image

from dualmem.encoders.base import EncoderCapability, EncoderConfig, l2_normalize


class ClipEncoder:
    """OpenAI CLIP via transformers."""
    name: str
    dim: int = 512
    capability = (
        EncoderCapability.ALIGNED | EncoderCapability.IMAGE_ONLY | EncoderCapability.TEXT_ONLY
    )

    def __init__(self, model: str = "openai/clip-vit-base-patch32",
                 config: Optional[EncoderConfig] = None):
        self.model_name = model
        self.name = f"clip:{model.split('/')[-1]}"
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
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._processor = CLIPProcessor.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._model.eval()
            try:
                self._model.to(self.config.device)
            except Exception:
                # GPU unavailable; stay on CPU
                self.config.device = "cpu"
            # Cache dims (CLIP-B/32 = 512, but be robust)
            self.dim = int(self._model.config.projection_dim)

    def embed_image(self, image_path: str) -> np.ndarray:
        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return l2_normalize(feats[0].cpu().numpy().astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return l2_normalize(feats[0].cpu().numpy().astype(np.float32))
