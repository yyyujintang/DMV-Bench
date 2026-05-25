"""OpenAI multimodal client (gpt-4o / gpt-4o-mini).

Same 4AFC interface as GeminiClient; reads images as base64 data URLs.
"""

from __future__ import annotations

import base64
import os
import time
from typing import List, Optional

from dualmem.vlm.base import VLMResponse, parse_choice


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 max_retries: int = 3, retry_base: float = 1.5,
                 base_url: Optional[str] = None):
        """`base_url` redirects to any OpenAI-compatible endpoint
        (vLLM, llama.cpp server, …). When set, api_key defaults to
        "EMPTY" since local endpoints don't auth."""
        from openai import OpenAI
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if self.base_url:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            self._client = OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_base = retry_base

    def _data_url(self, path: str) -> str:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"

    def four_afc(
        self,
        anchor_image_path: Optional[str],
        candidate_image_paths: List[str],
        instructions: str,
        extra_context_images: Optional[List[str]] = None,
    ) -> VLMResponse:
        content = [{"type": "text", "text": instructions}]
        if extra_context_images:
            for p in extra_context_images:
                content.append({"type": "image_url", "image_url": {"url": self._data_url(p)}})
        if anchor_image_path:
            content.append({"type": "text", "text": "ANCHOR IMAGE:"})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(anchor_image_path)}})
        content.append({"type": "text", "text": f"Candidates follow in order 0..{len(candidate_image_paths)-1}:"})
        for i, p in enumerate(candidate_image_paths):
            content.append({"type": "text", "text": f"Candidate {i}:"})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(p)}})
        content.append({"type": "text", "text": "Return exactly one line `Answer: N` with the matching index."})

        t0 = time.time()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=32,
                    temperature=0.0,
                )
                text = r.choices[0].message.content.strip()
                idx = parse_choice(text, n_candidates=len(candidate_image_paths))
                return VLMResponse(
                    text=text, chosen_index=idx,
                    latency_ms=int((time.time() - t0) * 1000),
                    raw={"model": self.model},
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
        content = [{"type": "text", "text": system_prompt + "\n\n" + user_text}]
        for p in [primary_image, *(extra_images or [])]:
            if not p:
                continue
            content.append({"type": "image_url", "image_url": {"url": self._data_url(p)}})
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"Action: done   # vlm error: {e}"
