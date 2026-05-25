"""VLM client interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class VLMResponse:
    text: str
    chosen_index: int          # -1 = parse failure
    latency_ms: int
    raw: Optional[dict] = None


class VLMClient(Protocol):
    """Uniform interface across all backbones (closed-source APIs + local OS)."""
    model: str

    def four_afc(
        self,
        anchor_image_path: Optional[str],
        candidate_image_paths: List[str],
        instructions: str,
        extra_context_images: Optional[List[str]] = None,
    ) -> VLMResponse: ...

    def generate_freeform(
        self,
        system_prompt: str,
        user_text: str,
        primary_image: Optional[str] = None,
        extra_images: Optional[List[str]] = None,
        max_tokens: int = 512,
    ) -> str: ...


# Shared parsing helper: extract integer 0..3 from a VLM's free text reply.
_NUM_PATTERNS = [
    re.compile(r"answer[:\s]*([0-9])", re.I),
    re.compile(r"choice[:\s]*([0-9])", re.I),
    re.compile(r"\b([0-9])\b\s*$", re.M),
    re.compile(r"^\s*([0-9])", re.M),
    re.compile(r"option\s*([0-9])", re.I),
    re.compile(r"index\s*[:=]?\s*([0-9])", re.I),
    re.compile(r"card\s*([0-9])", re.I),
]
_WORD_TO_DIGIT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "first": 0, "second": 1, "third": 2, "fourth": 3,
    "a": 0, "b": 1, "c": 2, "d": 3,
}


def parse_choice(text: str, n_candidates: int = 4) -> int:
    """Best-effort parse: pull a 0..n-1 integer from VLM's reply.

    Tries (in order): explicit "Answer: N", trailing digit, leading digit,
    "option/index/card N", word forms, single-letter A/B/C/D. Returns -1
    if nothing parseable.
    """
    if not text:
        return -1
    text_l = text.strip()
    for pat in _NUM_PATTERNS:
        m = pat.search(text_l)
        if m:
            try:
                idx = int(m.group(1))
                if 0 <= idx < n_candidates:
                    return idx
            except ValueError:
                pass
    # Word forms
    for w, d in _WORD_TO_DIGIT.items():
        if re.search(rf"\b{w}\b", text_l, re.I) and d < n_candidates:
            return d
    return -1
