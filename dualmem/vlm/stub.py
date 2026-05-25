"""Deterministic stub VLM (for tests + CI + offline pipeline validation).

Behavior: picks the candidate whose image-hash distance to the anchor is
smallest (sorted by file SHA). At fine grain tiers this won't be perfect
— which is what we want (the stub *should* show degradation across tiers
so we can validate that the runner records ceilings sensibly without
burning API tokens).
"""

from __future__ import annotations

import hashlib
import time
from typing import List, Optional

from dualmem.vlm.base import VLMResponse


class StubClient:
    def __init__(self, noise: float = 0.0, model: str = "stub-v0"):
        self.noise = noise
        self.model = model

    def _digest(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).digest()

    def four_afc(
        self,
        anchor_image_path: Optional[str],
        candidate_image_paths: List[str],
        instructions: str,
        extra_context_images: Optional[List[str]] = None,
    ) -> VLMResponse:
        t0 = time.time()
        if anchor_image_path is None:
            chosen = 0
        else:
            anchor_h = self._digest(anchor_image_path)
            scores = []
            for p in candidate_image_paths:
                cand_h = self._digest(p)
                # Hamming-ish distance on the first 32 bytes.
                dist = sum(bin(a ^ b).count("1") for a, b in zip(anchor_h[:32], cand_h[:32]))
                scores.append(dist)
            chosen = scores.index(min(scores))
        return VLMResponse(
            text=f"Answer: {chosen}",
            chosen_index=chosen,
            latency_ms=int((time.time() - t0) * 1000) + 1,
            raw={"stub": True},
        )

    def generate_freeform(
        self,
        system_prompt: str,
        user_text: str,
        primary_image: Optional[str] = None,
        extra_images: Optional[List[str]] = None,
        max_tokens: int = 512,
    ) -> str:
        # Deterministic agent reply for tests: navigate to chairs tier-1.
        return "Thought: stub agent\nAction: navigate(\"/category/chairs?tier=1\")"
