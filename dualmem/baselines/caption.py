"""Caption baseline: memory stores a VLM-generated caption of the anchor at encode time.

Stands in for M3-Agent caption-memory / HYMEM-caption.

The caption is produced by a real VLM call (cached on disk by anchor_slug).
This is the authentic injection-loss scenario: whatever the captioner chooses
to put into a 2-3 sentence description is all the memory has to discriminate
later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from dualmem.types import Trial


CAPTION_CACHE_PATH = "data/vismem_diag/_caption_cache.json"
_CAPTIONER = None
_CACHE = None


def _load_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = Path(CAPTION_CACHE_PATH)
    if p.exists():
        _CACHE = json.loads(p.read_text())
    else:
        _CACHE = {}
    return _CACHE


def _save_cache():
    if _CACHE is None: return
    p = Path(CAPTION_CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_CACHE, indent=2))


def _get_captioner():
    """Lazy-init a Gemini Flash captioner. Set DUALMEM_CAPTIONER=stub
    to use a deterministic synthesized caption instead (for tests).
    """
    global _CAPTIONER
    if _CAPTIONER is not None: return _CAPTIONER
    if os.environ.get("DUALMEM_CAPTIONER") == "stub":
        _CAPTIONER = "stub"
        return _CAPTIONER
    from dualmem.vlm.gemini import GeminiClient
    _CAPTIONER = GeminiClient(model="gemini-2.5-flash")
    return _CAPTIONER


CAPTION_PROMPT = (
    "Describe this product photograph in 2 to 3 sentences. Focus on attributes that would help "
    "someone identify this exact product later (color, shape, material, distinctive features). "
    "Do not invent attributes not visible in the image. Do not number, do not include preamble."
)


def _caption_for(trial: Trial) -> str:
    """VLM-generated caption of the anchor image. Cached by slug+tier."""
    key = f"{trial.anchor_slug}_t{trial.cell.grain_tier}"
    cache = _load_cache()
    if key in cache:
        return cache[key]
    cap = _captioner_call(trial.anchor_path)
    cache[key] = cap
    _save_cache()
    return cap


def _captioner_call(image_path: str) -> str:
    cap = _get_captioner()
    if cap == "stub":
        # Deterministic fallback for CI / tests.
        return f"<stub caption for {image_path}>"
    # Reuse the four_afc API: pass anchor image and ask for a free-text caption.
    # We hack this by using 0 candidates and an open-ended instruction.
    from dualmem.vlm.base import VLMResponse
    # Direct generate_content call avoids the four_afc parsing machinery.
    from google.genai import types
    with open(image_path, "rb") as f:
        data = f.read()
    part = types.Part.from_bytes(data=data, mime_type="image/png")
    resp = cap._client.models.generate_content(
        model=cap.model,
        contents=[CAPTION_PROMPT, part],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (resp.text or "").strip()


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class Caption:
    name = "Caption"

    def __init__(self):
        self._mem: List[Trial] = []
        self._captions: dict = {}

    def reset(self):
        self._mem = []
        self._captions = {}

    def encode(self, trial: Trial):
        self._mem.append(trial)
        self._captions[trial.anchor_slug] = _caption_for(trial)

    def retrieve(self, query_text: str) -> Optional[Trial]:
        if not self._mem:
            return None
        q = _tokens(query_text)
        scored = [(len(q & _tokens(self._captions[m.anchor_slug])), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def oracle_inject(self, trial: Trial) -> dict:
        cap = self._captions.get(trial.anchor_slug) or _caption_for(trial)
        return {"text": f"Memory caption (visual channel summarized as text):\n\"{cap}\"", "images": []}
