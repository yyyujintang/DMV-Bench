"""Qwen-Image (text-to-image) backend.

Wraps the diffusers `QwenImagePipeline` for `Qwen/Qwen-Image` — Qwen's
text-to-image model, complementary to the edit variant in
`qwen_image_edit.py`. Use this when you want a fully self-hosted t2i
path (no Gemini API spend).

Typical chained workflow:
  Qwen-Image (t2i)  →  seed photo
      ↓ feeds source_image into
  Qwen-Image-Edit-2509 (edit)  →  120 variants

The two backends are deliberately separate classes (rather than a
single multi-mode class) because the diffusers pipeline classes
differ; one HF download per pipeline; resident memory ~50 GB each
when loaded.

Reference (verified 2026-05-15):
  - HF model id   : Qwen/Qwen-Image           (released 2025-08-18, 174k downloads)
  - diffusers ver : 0.36.0                    (ships QwenImagePipeline)
  - transformers  : 4.57.6
  - torch         : 2.10.0+cu128
"""
from __future__ import annotations

import io
from typing import Any

from .base import (
    BackendError,
    BackendUnsupportedError,
    GenerateRequest,
)


DEFAULT_MODEL = "Qwen/Qwen-Image"


class QwenImageBackend:
    """ImageBackend backed by Qwen's text-to-image diffusion model."""

    capabilities = frozenset({"text2image"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
    ):
        try:
            import torch
            from diffusers import QwenImagePipeline
        except ImportError as e:
            raise BackendError(
                f"Qwen-Image requires diffusers + torch with CUDA. "
                f"Import failure: {e}"
            ) from e
        self.model_id = model_id
        self.model_revision = revision or "main"
        self.backend_id = "qwen-image"
        self._dtype_str = dtype
        self._device = device
        try:
            self._dtype = getattr(torch, dtype)
        except AttributeError as e:
            raise BackendError(f"unknown torch dtype: {dtype!r}") from e
        self._pipe = QwenImagePipeline.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=self._dtype,
        ).to(self._device)
        try:
            self._pipe.set_progress_bar_config(disable=True)
        except AttributeError:
            pass
        self._steps = num_inference_steps
        self._cfg = true_cfg_scale
        self._width = width
        self._height = height

    def generate(self, req: GenerateRequest) -> bytes:
        if req.mode != "text2image":
            raise BackendUnsupportedError(
                "qwen-image only supports mode='text2image'; for edit "
                "mode use the QwenImageEditBackend"
            )
        import torch
        gen = torch.Generator(self._device).manual_seed(
            (req.seed or 0) & 0x7fffffff
        )
        try:
            # Pass negative_prompt only when supplied — without it the
            # diffusers pipeline emits a warning that classifier-free
            # guidance is disabled, and the prompt's "no shelving" /
            # "no frame" clauses get weakly enforced.
            kwargs: dict[str, Any] = {
                "prompt": req.prompt,
                "width": self._width,
                "height": self._height,
                "num_inference_steps": self._steps,
                "true_cfg_scale": self._cfg,
                "generator": gen,
            }
            if req.negative_prompt:
                kwargs["negative_prompt"] = req.negative_prompt
            result = self._pipe(**kwargs)
        except RuntimeError as e:
            raise BackendError(f"qwen pipeline runtime error: {e}") from e
        if not getattr(result, "images", None):
            raise BackendError("qwen pipeline returned no images")
        out_img = result.images[0]
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return buf.getvalue()
