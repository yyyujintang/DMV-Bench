"""Post-generation QC: CLIP text-image cosine similarity.

A passing image should embed close to its own prompt text. We use the
same model that powers `tasks/cache/clip_v1.json` (ViT-L/14, 768-dim)
so the QC threshold is consistent with downstream uniqueness checks.

Threshold:
  CLIP_TEXT_THRESHOLD = 0.25  — matching prompts typically score
  0.25-0.35 with CLIP L/14. Random text scores < 0.20. The threshold
  is a sanity floor, not a quality maximiser.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

CLIP_TEXT_THRESHOLD = 0.25
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


_model = None
_processor = None
_device = None


def _ensure_clip() -> None:
    """Lazy-load CLIP. Caller is single-threaded so no locking needed."""
    global _model, _processor, _device
    if _model is not None:
        return
    import torch  # local import — avoids torch at module load
    from transformers import CLIPModel, CLIPProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_device).eval()
    _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


def clip_text_image_cosine(image_bytes: bytes, prompt: str) -> float:
    """Return cos(clip_image(img), clip_text(prompt)) ∈ [-1, 1].

    Truncates the prompt to CLIP's 77-token context (transformers tokenizer
    handles this internally with truncation=True).
    """
    _ensure_clip()
    import torch
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(  # type: ignore[union-attr]
        text=[prompt],
        images=[img],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(_device)
    with torch.no_grad():
        out = _model(**inputs)  # type: ignore[union-attr]
    img_emb = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
    cos = (img_emb * txt_emb).sum(dim=-1).item()
    return float(cos)
