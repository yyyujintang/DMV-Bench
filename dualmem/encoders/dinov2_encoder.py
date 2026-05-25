"""DINOv2 encoder — image-only, no aligned text encoder.

Stub for now; full inference is reserved for GPU machines. The interface
is identical to ClipEncoder so retrievers can swap encoders without code
change. Calling `embed_image` from this stub raises a NotImplementedError
unless `eager=True` was passed (which actually triggers the (slow) CPU
inference path).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from PIL import Image

from dualmem.encoders.base import EncoderCapability, EncoderConfig, l2_normalize


class Dinov2Encoder:
    name: str
    dim: int = 1024     # ViT-L
    capability = EncoderCapability.IMAGE_ONLY

    def __init__(self, model: str = "facebook/dinov2-large",
                 config: Optional[EncoderConfig] = None,
                 eager: bool = False):
        self.model_name = model
        self.name = f"dinov2:{model.split('/')[-1]}"
        self.config = config or EncoderConfig()
        self.eager = eager
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.eager:
                raise NotImplementedError(
                    f"{self.name} is a placeholder. Pass eager=True at construction "
                    "to run real CPU inference (slow on ViT-L). Recommended path: "
                    "run on a GPU box."
                )
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir
            )
            self._model.eval()
            try:
                self._model.to(self.config.device)
            except Exception:
                self.config.device = "cpu"
            self.dim = int(self._model.config.hidden_size)

    def embed_image(self, image_path: str) -> np.ndarray:
        self._ensure_loaded()
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            out = self._model(**inputs)
        # CLS token / pooled output
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output[0]
        else:
            feats = out.last_hidden_state[0, 0]
        return l2_normalize(feats.cpu().numpy().astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "DINOv2 is image-only and has no aligned text encoder. "
            "Pair it with an SbertEncoder for text retrieval."
        )
