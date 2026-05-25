"""Gemini 2.5 Flash Image (Nano Banana) wrapper.

Two operations we need:
    generate_reference(prompt) -> PIL.Image     # photo-realistic base photo
    edit_to_variant(ref_image, edit_prompt, seed=None) -> PIL.Image

The Nano Banana model accepts a text prompt OR (image + text) input and
returns inline image bytes. Latency is ~3-6 seconds per call.

To preserve product *identity* across variants (same silhouette/material,
only color/finish changes) we always feed the *reference* image plus a
strict edit instruction. This is the key to a controlled visual-granularity
ladder: variant_b at tier_1 differs from variant_a only on color, NOT
on layout or pose.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


REFERENCE_PROMPT_TEMPLATE = (
    "A professional product catalog photograph of a {description}. "
    "Centered subject on a pure clean studio background, soft even lighting, "
    "no logos, no text, no people, no props. Realistic high-resolution e-commerce style. "
    "Square aspect ratio."
)

EDIT_PROMPT_TEMPLATE = (
    "Take this product photograph and produce a single edited version where the only change is "
    "the {attribute}. The product silhouette, material weave, pose, framing, "
    "lighting, and background must be IDENTICAL to the input. {delta_instruction} "
    "Do not add or remove decorative elements. Output one square e-commerce-style image."
)


class NanoBanana:
    """Wraps gemini-2.5-flash-image (Nano Banana) for product image generation."""

    def __init__(self, model: str = "gemini-2.5-flash-image", api_key: Optional[str] = None,
                 max_retries: int = 3, retry_base: float = 1.5):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_base = retry_base

    def _extract_image(self, resp) -> Optional[Image.Image]:
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                if getattr(part, "inline_data", None) is not None and part.inline_data.data:
                    return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        return None

    def _call(self, contents) -> Image.Image:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents,
                )
                img = self._extract_image(resp)
                if img is not None:
                    return img
                last_err = RuntimeError("no image part in response")
            except Exception as e:
                last_err = e
            time.sleep(self.retry_base ** attempt)
        raise RuntimeError(f"NanoBanana failed after {self.max_retries} attempts: {last_err}")

    def generate_reference(self, description: str) -> Image.Image:
        prompt = REFERENCE_PROMPT_TEMPLATE.format(description=description)
        return self._call(prompt)

    def edit_to_variant(self, ref_image_path: str, attribute: str, delta_instruction: str) -> Image.Image:
        prompt = EDIT_PROMPT_TEMPLATE.format(attribute=attribute, delta_instruction=delta_instruction)
        with open(ref_image_path, "rb") as f:
            ref_bytes = f.read()
        part = self._types.Part.from_bytes(data=ref_bytes, mime_type="image/png")
        return self._call([prompt, part])


def save_image(img: Image.Image, path: str, size: int = 512) -> str:
    """Resize to a square, save as PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = img.resize((size, size), Image.LANCZOS)
    img.save(p, format="PNG")
    return str(p)
