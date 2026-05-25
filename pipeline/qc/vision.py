"""Post-generation QC: detail-presence vision check via Gemini.

Uses **gemini-2.5-flash** (text+vision VLM, NOT the image-generation
model) to answer a single yes/no question about whether the required
incidental detail is visible in the generated image.

Prompt is a strict yes/no so the answer parser is trivial.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request, error

VISION_MODEL = "gemini-2.5-flash"
VISION_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL}:generateContent"
)


@dataclass
class VisionCheckResult:
    detail_phrase: str
    answer: str            # raw stripped reply, e.g. "Yes" or "No"
    passed: bool           # parsed to a bool
    raw: dict[str, Any]    # full Gemini response for debugging


def detail_visible(image_bytes: bytes, detail_phrase: str,
                   api_key: str | None = None,
                   timeout: float = 30.0) -> VisionCheckResult:
    """Ask Gemini whether the detail is visible in the image.

    Question template kept terse — the LLM should reply exactly
    'Yes' or 'No', not an essay. We accept any answer that starts
    with 'yes' (case-insensitive) as a pass.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set for vision QC")
    question = (
        f"Look at this product image. Does it clearly show "
        f"{detail_phrase}? Answer with one word: Yes or No."
    )
    body = json.dumps({
        "contents": [{
            "role": "user",
            "parts": [
                {"text": question},
                {"inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
            ],
        }],
    }).encode("utf-8")
    req = request.Request(
        VISION_ENDPOINT,
        data=body,
        headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"vision QC HTTP {e.code}: {body_text}") from e
    text = ""
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if isinstance(part.get("text"), str):
                text = part["text"].strip()
                break
        if text:
            break
    passed = text.lower().startswith("yes")
    return VisionCheckResult(
        detail_phrase=detail_phrase,
        answer=text,
        passed=passed,
        raw={"first200": json.dumps(payload)[:200]},
    )
