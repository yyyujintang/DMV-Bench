"""Gemini VLM client (google-genai SDK).

Wraps the multi-image input pattern used by the 4AFC prompt:
  [Instructions text] [Anchor image] [Candidate 0 image] ... [Candidate 3 image]
The model is asked to return a single integer 0..3.

Robustness:
- Retries on transient API failures (up to 3 attempts with exponential backoff).
- Caches a single client across calls (created lazily on first use).
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from dualmem.vlm.base import VLMResponse, parse_choice


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None,
                 max_retries: int = 3, retry_base: float = 1.5):
        from google import genai
        from google.genai import types as gtypes

        self.model = model
        self._genai = genai
        self._gtypes = gtypes
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self._client = genai.Client(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_base = retry_base

    def _read_image_part(self, path: str):
        # Gemini accepts inline bytes with explicit mime type. PNG only for now.
        with open(path, "rb") as f:
            data = f.read()
        return self._gtypes.Part.from_bytes(data=data, mime_type="image/png")

    def four_afc(
        self,
        anchor_image_path: Optional[str],
        candidate_image_paths: List[str],
        instructions: str,
        extra_context_images: Optional[List[str]] = None,
    ) -> VLMResponse:
        parts = [instructions]
        if extra_context_images:
            for p in extra_context_images:
                parts.append(self._read_image_part(p))
        if anchor_image_path:
            parts.append("ANCHOR IMAGE (the one you must match):")
            parts.append(self._read_image_part(anchor_image_path))
        parts.append(f"\nThe {len(candidate_image_paths)} candidates follow in order 0..{len(candidate_image_paths)-1}:")
        for i, p in enumerate(candidate_image_paths):
            parts.append(f"Candidate {i}:")
            parts.append(self._read_image_part(p))
        parts.append(
            f"\nReturn exactly one line of the form `Answer: N` where N is the integer index "
            f"(0..{len(candidate_image_paths)-1}) of the candidate that matches the anchor. "
            f"Do not output any explanation."
        )

        t0 = time.time()
        last_err = None
        # Gemini 2.5 Flash spends tokens on internal reasoning before output;
        # disabling thinking eliminates the truncation that was hitting the
        # tight token budget and returning bare "Answer" / "Answer:" prefixes.
        # If a future variant *needs* thinking, set a generous budget instead.
        cfg_kwargs = dict(temperature=0.0, max_output_tokens=256)
        if "2.5" in self.model:
            cfg_kwargs["thinking_config"] = self._gtypes.ThinkingConfig(thinking_budget=0)
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=self._gtypes.GenerateContentConfig(**cfg_kwargs),
                )
                text = (resp.text or "").strip()
                idx = parse_choice(text, n_candidates=len(candidate_image_paths))
                return VLMResponse(
                    text=text, chosen_index=idx,
                    latency_ms=int((time.time() - t0) * 1000),
                    raw={"model": self.model, "attempt": attempt},
                )
            except Exception as e:
                last_err = e
                time.sleep(self.retry_base ** attempt)
        return VLMResponse(
            text=f"<ERROR: {type(last_err).__name__}: {last_err}>",
            chosen_index=-1,
            latency_ms=int((time.time() - t0) * 1000),
            raw={"model": self.model, "error": str(last_err)},
        )

    # ------------------------------------------------------------------
    # Free-form generation (used by the Playwright agent loop)
    # ------------------------------------------------------------------
    def generate_freeform(
        self,
        system_prompt: str,
        user_text: str,
        primary_image: Optional[str] = None,
        extra_images: Optional[List[str]] = None,
        max_tokens: int = 512,
    ) -> str:
        gtypes = self._gtypes
        parts: list = [system_prompt, user_text]
        for ip in [primary_image, *(extra_images or [])]:
            if ip:
                with open(ip, "rb") as f:
                    parts.append(gtypes.Part.from_bytes(data=f.read(), mime_type="image/png"))
        cfg_kwargs = dict(temperature=0.0, max_output_tokens=max_tokens)
        if "2.5" in self.model:
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=parts,
                config=gtypes.GenerateContentConfig(**cfg_kwargs),
            )
            return (resp.text or "").strip()
        except Exception as e:
            return f"Action: done   # vlm error: {e}"
