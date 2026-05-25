"""Qwen-Image-Edit backend (Phase B).

Defaults to the newer **Qwen-Image-Edit-2509** ("Plus") variant on
HuggingFace, paired with `QwenImageEditPlusPipeline`. The older
`Qwen/Qwen-Image-Edit` (paired with `QwenImageEditPipeline`) is still
supported by passing `model_id="Qwen/Qwen-Image-Edit"` + `pipeline_class="edit"`.

Edit-only; raises `BackendUnsupportedError` if asked to do text-to-image.

The diffusers + torch imports happen inside `__init__` so a Gemini-only
run never pays the cost. On the the HPC cluster, the login node has no
GPU — instantiate this backend only from inside a SLURM job
(`scripts/run_qwen_edit.sbatch`).

Reference (verified 2026-05-15):
  - latest model  : Qwen/Qwen-Image-Edit-2509   (released 2025-09-22, 212k downloads)
  - older model   : Qwen/Qwen-Image-Edit        (released 2025-08-25, 67k downloads)
  - diffusers ver : 0.36.0
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


# Default to the newer 2509 ("Plus") variant. Override via kwargs to
# pick the older Qwen/Qwen-Image-Edit + pipeline_class="edit".
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_PIPELINE_CLASS = "plus"


class QwenImageEditBackend:
    """ImageBackend backed by Qwen's instruction-conditioned edit model."""

    capabilities = frozenset({"edit"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        pipeline_class: str = DEFAULT_PIPELINE_CLASS,  # "edit" | "plus"
    ):
        try:
            import torch
            from diffusers import (
                QwenImageEditPipeline,
                QwenImageEditPlusPipeline,
            )
        except ImportError as e:
            raise BackendError(
                f"Qwen-Image-Edit requires diffusers + torch with CUDA. "
                f"Import failure: {e}"
            ) from e
        self.model_id = model_id
        self.model_revision = revision or "main"
        self.backend_id = (
            f"qwen-image-edit-plus" if pipeline_class == "plus"
            else "qwen-image-edit"
        )
        self._dtype_str = dtype
        self._device = device
        try:
            self._dtype = getattr(torch, dtype)
        except AttributeError as e:
            raise BackendError(f"unknown torch dtype: {dtype!r}") from e
        if pipeline_class == "plus":
            cls: Any = QwenImageEditPlusPipeline
        elif pipeline_class == "edit":
            cls = QwenImageEditPipeline
        else:
            raise ValueError(
                f"pipeline_class must be 'edit' or 'plus', got {pipeline_class!r}"
            )
        self._pipe = cls.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=self._dtype,
        ).to(self._device)
        # Disable progress bars; we log per-image at the driver level.
        try:
            self._pipe.set_progress_bar_config(disable=True)
        except AttributeError:
            pass
        self._steps = num_inference_steps
        self._cfg = true_cfg_scale

    def generate(self, req: GenerateRequest) -> bytes:
        if req.mode != "edit":
            raise BackendUnsupportedError(
                "qwen-image-edit only supports mode='edit'; "
                "pass --source existing-file or --source backend"
            )
        if req.source_image is None:
            # Defensive — should be caught by mode check above.
            raise BackendUnsupportedError(
                "edit mode requires a non-None source_image"
            )
        import torch
        from PIL import Image
        src = Image.open(io.BytesIO(req.source_image)).convert("RGB")
        gen = torch.Generator(self._device).manual_seed(
            (req.seed or 0) & 0x7fffffff
        )
        try:
            kwargs: dict[str, Any] = {
                "image": src,
                "prompt": req.prompt,
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
