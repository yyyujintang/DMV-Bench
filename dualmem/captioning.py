"""Shared VLM captioner — a caption_fn factory consumed by MemorySystem
constructions that need a VLM-generated caption (Caption, DualChannel,
HYMEM).

The captioner is cached on disk by `(slug or basename, tier?)`. Tests
can pass `caption_fn=stub_caption_fn` to skip API calls entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional


CAPTION_PROMPT = (
    "Describe this product photograph in 2 to 3 sentences. Focus on attributes that would help "
    "someone identify this exact product later (color, shape, material, distinctive features). "
    "Do not invent attributes not visible in the image. Do not number, do not include preamble."
)
# The captioner used for every result in the paper. It is deliberately steered
# at product attributes and is NOT filtered against the cue vocabulary: whether
# a cue survives compression into language is a measured property of the
# baseline, not something the harness enforces.
DEFAULT_CACHE = "data/vismem_diag_v2/_caption_cache.json"

# An unsteered, exhaustive alternative: describes everything visible rather
# than product attributes, so it verbalises the incidental cue far more often.
# Used to show the DualMem lead is not an artefact of a stingy captioner. It
# must never hint that cues exist -- that would leak the benchmark's signal.
EXHAUSTIVE_CAPTION_PROMPT = (
    "Describe this photograph in detail in 3 to 5 sentences. Describe every object visible in "
    "the image, including the setting, the background, and any smaller items present, together "
    "with their colors and materials. Do not invent things not visible in the image. "
    "Do not number, do not include preamble."
)
EXHAUSTIVE_CACHE = "data/vismem_diag_v2/_caption_cache_exhaustive.json"


def _cache_key(image_path: str) -> str:
    """Key by the path components AFTER 'images/' — stable across machines AND
    distinguishing the base / with_cue / online_cue variant of a product.
    e.g.  base/chair/modern/00.png   vs   with_cue/chair/modern/00.png
    (keying by `<cat>/<style>/<idx>.png` alone collides base with with_cue —
    a base image would wrongly hit the with_cue caption)."""
    parts = Path(image_path).parts
    if "images" in parts:
        return "/".join(parts[parts.index("images") + 1:])
    return "/".join(parts[-3:]) if len(parts) >= 3 else os.path.basename(image_path)


def make_gemini_caption_fn(model: str = "gemini-2.5-flash",
                           cache_path: str = DEFAULT_CACHE,
                           prompt: str = CAPTION_PROMPT) -> Callable[[str], str]:
    """Return a `caption_fn(image_path) -> str` that calls Gemini and
    caches on disk."""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    # Lazily build the Gemini client only when actually needed.
    _state = {"client": None, "types": None}

    def _client():
        if _state["client"] is None:
            from dualmem.vlm.gemini import GeminiClient
            from google.genai import types
            _state["client"] = GeminiClient(model=model)
            _state["types"] = types
        return _state["client"], _state["types"]

    def caption_fn(image_path: str) -> str:
        key = _cache_key(image_path)
        if key in cache:
            return cache[key]
        cli, gtypes = _client()
        with open(image_path, "rb") as f:
            data = f.read()
        # Retry on transient 503 / rate-limit / network failures.
        import time as _time
        last_err = None
        for attempt in range(3):
            try:
                part = gtypes.Part.from_bytes(data=data, mime_type="image/png")
                resp = cli._client.models.generate_content(
                    model=cli.model,
                    contents=[prompt, part],
                    config=gtypes.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=200,
                        thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
                    ),
                )
                text = (resp.text or "").strip()
                cache[key] = text
                # Atomic write (temp + rename) — safe when several baseline
                # processes share this cache file under the parallel launcher.
                tmp = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(cache, indent=2))
                tmp.replace(cache_file)
                return text
            except Exception as e:
                last_err = e
                _time.sleep(2.0 * (attempt + 1))
        # Final fallback: return a stub caption so the agent run continues.
        return f"<caption error: {type(last_err).__name__}>"

    return caption_fn


def stub_caption_fn(image_path: str) -> str:
    """Deterministic stub for tests."""
    base = os.path.basename(image_path)
    h = hashlib.sha1(image_path.encode()).hexdigest()[:6]
    return f"<stub caption {h} for {base}>"


def make_exhaustive_caption_fn(model: str = "gemini-2.5-flash",
                               cache_path: str = EXHAUSTIVE_CACHE) -> Callable[[str], str]:
    """The unsteered captioner, with its OWN cache file -- the two prompts must
    never share cache entries."""
    return make_gemini_caption_fn(model=model, cache_path=cache_path,
                                  prompt=EXHAUSTIVE_CAPTION_PROMPT)
