"""Anthropic Claude multimodal client.

Default model: Claude 3.7 Sonnet (`claude-3-7-sonnet-20250219`).

Same `four_afc` + `generate_freeform` interface as the other clients in this
package. Reads images as base64-encoded `image/png` blocks.
"""

from __future__ import annotations

import base64
import os
import time
from typing import List, Optional

from dualmem.vlm.base import VLMResponse, parse_choice


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base: float = 1.5,
    ):
        from anthropic import Anthropic

        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self._client = Anthropic(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_base = retry_base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _image_block(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = f.read()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(data).decode(),
            },
        }

    @staticmethod
    def _text_block(text: str) -> dict:
        return {"type": "text", "text": text}

    def _create_with_retry(self, messages: list, max_tokens: int) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    messages=messages,
                )
                # `content` is a list of content blocks; join the text ones.
                chunks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                return ("".join(chunks)).strip()
            except Exception as e:
                last_err = e
                time.sleep(self.retry_base ** attempt)
        raise last_err if last_err else RuntimeError("anthropic call failed")

    # ------------------------------------------------------------------
    # 4AFC
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
            content.append(self._text_block("ANCHOR IMAGE (the one you must match):"))
            content.append(self._image_block(anchor_image_path))
        content.append(self._text_block(
            f"Candidates follow in order 0..{len(candidate_image_paths)-1}:"
        ))
        for i, p in enumerate(candidate_image_paths):
            content.append(self._text_block(f"Candidate {i}:"))
            content.append(self._image_block(p))
        content.append(self._text_block(
            f"Return exactly one line `Answer: N` where N in 0..{len(candidate_image_paths)-1}."
        ))

        t0 = time.time()
        try:
            text = self._create_with_retry(
                [{"role": "user", "content": content}],
                max_tokens=32,
            )
            idx = parse_choice(text, n_candidates=len(candidate_image_paths))
            return VLMResponse(
                text=text,
                chosen_index=idx,
                latency_ms=int((time.time() - t0) * 1000),
                raw={"model": self.model},
            )
        except Exception as e:
            return VLMResponse(
                text=f"<ERROR: {type(e).__name__}: {e}>",
                chosen_index=-1,
                latency_ms=int((time.time() - t0) * 1000),
                raw={"model": self.model, "error": str(e)},
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
        content: list = [self._text_block(user_text)]
        for p in [primary_image, *(extra_images or [])]:
            if p:
                content.append(self._image_block(p))
        try:
            return self._create_with_retry(
                [
                    # Claude takes a top-level `system` parameter in
                    # `.messages.create` — we route the system prompt into
                    # the user turn for parity with the other clients and to
                    # avoid an SDK-specific param shape. The agent's system
                    # prompt is small so the cost is negligible.
                    {
                        "role": "user",
                        "content": [self._text_block(f"[SYSTEM]\n{system_prompt}\n\n[USER]\n"), *content],
                    }
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            return f"Action: done   # vlm error: {e}"
