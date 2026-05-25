"""Qwen2.5-VL local-inference client (3B / 7B Instruct).

Loads `Qwen/Qwen2.5-VL-{3B,7B}-Instruct` via transformers, keeps the model
+ processor cached at module level so re-instantiating `QwenVLClient` in the
same process doesn't reload weights.

Defaults:
- bf16 on GPU (`cuda`), fp32 on CPU (CPU is for smoke tests only — 7B in fp32
  on CPU will be ~28 GB and slow).
- `device_map="auto"` so multi-GPU machines shard naturally.
- max_pixels capped so very large catalog images don't blow up the visual
  token budget; the catalog images we generate are 1024x1024 which fits.

Used for: large-scale ablation runs in the open-source slot of the main
table. See `doc/vlm_backends.md` for the backbone-coverage policy.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from dualmem.vlm.base import VLMResponse, parse_choice


# Map our friendly backend names to HF repo IDs.
QWEN_VL_MODEL_IDS = {
    "qwen-vl-3b":          "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen-vl-7b":          "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b":       "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b":       "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}


# Module-level cache: one (model, processor) per repo id.
_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}
_CACHE_LOCK = Lock()


def _resolve_repo_id(name: str) -> str:
    if name in QWEN_VL_MODEL_IDS:
        return QWEN_VL_MODEL_IDS[name]
    # Permit a custom repo id we don't know about.
    if "/" in name:
        return name
    raise ValueError(
        f"Unknown Qwen-VL model alias: {name!r}. "
        f"Known: {sorted(set(QWEN_VL_MODEL_IDS))}"
    )


def _load(repo_id: str, device: str, dtype_name: str, max_pixels: int):
    """Idempotent load. Returns (model, processor)."""
    cache_key = f"{repo_id}|{device}|{dtype_name}|{max_pixels}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[dtype_name]

        # device_map="auto" works on GPU; on pure CPU we pass None and place
        # the model manually to avoid an accelerate dispatch on a 1-device
        # setup.
        load_kwargs = {"torch_dtype": torch_dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, **load_kwargs)
        if device == "cpu":
            model = model.to("cpu")
        model.eval()

        # max_pixels here is the *upper* bound on the visual encoder input.
        # Setting it on the processor caps the token budget per image; this
        # matters at 1024x1024 where the default can produce ~1.6k vision
        # tokens per image and inflate prefill latency 3-4x on 7B.
        processor = AutoProcessor.from_pretrained(
            repo_id,
            min_pixels=256 * 28 * 28,
            max_pixels=max_pixels,
        )

        _MODEL_CACHE[cache_key] = (model, processor)
        return model, processor


class QwenVLClient:
    """Local Qwen2.5-VL client. Lazy-loads model on first call."""

    def __init__(
        self,
        model: str = "qwen-vl-7b",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_pixels: int = 1280 * 28 * 28,  # ~1.0M pixels; fine for 1024x1024 catalog images
        max_new_tokens: int = 256,
    ):
        self.repo_id = _resolve_repo_id(model)
        # Keep `model` as the canonical short name (used by run-id stamping
        # and by playwright_agent's "model name" introspection).
        self.model = self.repo_id.split("/")[-1]

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.dtype = dtype
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        # Lazy-load: weights only materialize on first request so that an
        # un-imported QwenVLClient instantiation (e.g. in __init__.py
        # factory dispatch under stub mode) doesn't pay the load cost.
        self._loaded = False
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._loaded:
            return
        self._model, self._processor = _load(
            self.repo_id, self.device, self.dtype, self.max_pixels
        )
        self._loaded = True

    @staticmethod
    def _image_block(path: str) -> dict:
        # Qwen's chat-template accepts `file://` URIs as well as PIL Images.
        # Absolute path → file:// URI keeps the chat template parser happy.
        p = Path(path).resolve()
        return {"type": "image", "image": f"file://{p}"}

    @staticmethod
    def _text_block(text: str) -> dict:
        return {"type": "text", "text": text}

    # ------------------------------------------------------------------
    def _generate(self, messages: List[dict], max_new_tokens: Optional[int] = None) -> str:
        self._ensure_loaded()
        from qwen_vl_utils import process_vision_info

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to the model's first device.
        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        import torch
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        # Strip the prompt prefix.
        in_len = inputs["input_ids"].shape[-1]
        trimmed = out[:, in_len:]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    # ------------------------------------------------------------------
    def four_afc(
        self,
        anchor_image_path: Optional[str],
        candidate_image_paths: List[str],
        instructions: str,
        extra_context_images: Optional[List[str]] = None,
    ) -> VLMResponse:
        content: list = [self._text_block(instructions)]
        if extra_context_images:
            for p in extra_context_images:
                content.append(self._image_block(p))
        if anchor_image_path:
            content.append(self._text_block("ANCHOR IMAGE (match this):"))
            content.append(self._image_block(anchor_image_path))
        content.append(self._text_block(
            f"Candidates 0..{len(candidate_image_paths)-1}:"
        ))
        for i, p in enumerate(candidate_image_paths):
            content.append(self._text_block(f"Candidate {i}:"))
            content.append(self._image_block(p))
        content.append(self._text_block(
            f"Return exactly one line `Answer: N` (N in 0..{len(candidate_image_paths)-1})."
        ))

        messages = [{"role": "user", "content": content}]

        t0 = time.time()
        try:
            text = self._generate(messages, max_new_tokens=32)
            idx = parse_choice(text, n_candidates=len(candidate_image_paths))
            return VLMResponse(
                text=text,
                chosen_index=idx,
                latency_ms=int((time.time() - t0) * 1000),
                raw={"model": self.model, "repo_id": self.repo_id},
            )
        except Exception as e:
            return VLMResponse(
                text=f"<ERROR: {type(e).__name__}: {e}>",
                chosen_index=-1,
                latency_ms=int((time.time() - t0) * 1000),
                raw={"model": self.model, "error": str(e)},
            )

    # ------------------------------------------------------------------
    def generate_freeform(
        self,
        system_prompt: str,
        user_text: str,
        primary_image: Optional[str] = None,
        extra_images: Optional[List[str]] = None,
        max_tokens: int = 512,
    ) -> str:
        # Qwen-VL's chat template supports a dedicated `system` role.
        messages = [
            {"role": "system", "content": [self._text_block(system_prompt)]},
        ]
        user_content: list = [self._text_block(user_text)]
        for p in [primary_image, *(extra_images or [])]:
            if p:
                user_content.append(self._image_block(p))
        messages.append({"role": "user", "content": user_content})
        try:
            return self._generate(messages, max_new_tokens=max_tokens)
        except Exception as e:
            return f"Action: done   # vlm error: {e}"
