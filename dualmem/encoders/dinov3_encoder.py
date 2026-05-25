"""DINOv3 encoder — image-only, latest Meta backbone.

Same interface as Dinov2Encoder. If a local checkpoint path is available
(env var `DINOV3_LOCAL_PATH`), it is preferred over HuggingFace Hub.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
from PIL import Image

from dualmem.encoders.base import EncoderCapability, EncoderConfig, l2_normalize


DEFAULT_DINOV3_PATH = "$DATA_ROOT/models/dinov3/dinov3-vitl16-pretrain-lvd1689m"


class Dinov3Encoder:
    name: str
    dim: int = 1024
    capability = EncoderCapability.IMAGE_ONLY

    def __init__(self, model: Optional[str] = None,
                 config: Optional[EncoderConfig] = None,
                 eager: bool = False):
        # Prefer a local path if it exists, else use the HF Hub model id.
        local_path = os.environ.get("DINOV3_LOCAL_PATH", DEFAULT_DINOV3_PATH)
        if model is None:
            model = local_path if os.path.isdir(local_path) else "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.model_name = model
        self.name = f"dinov3:{os.path.basename(model)}"
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
                    "to run real CPU inference. Recommended path: GPU machine."
                )
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.config.cache_dir, trust_remote_code=True
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
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output[0]
        else:
            feats = out.last_hidden_state[0, 0]
        return l2_normalize(feats.cpu().numpy().astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "DINOv3 is image-only. Pair with an SbertEncoder for text retrieval."
        )
