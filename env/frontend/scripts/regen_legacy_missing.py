"""One-off: regenerate 48 images (4 categories × 4 styles × 3 tiers) for
the categories that had no legacy_colors backup, using the same prompt
shape that generated the original 72 backed-up images.

Targets: sofa / bookshelf / plant_pot / wall_art.

Usage:  GEMINI_API_KEY=... python scripts/regen_legacy_missing.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_IMAGES = ROOT / "public" / "images"

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# 4 categories that lacked legacy_colors backup. Singular noun forms match
# the directory naming convention (`public/images/<dir>/`).
CATEGORIES = {
    "sofa":      "two-seat sofa",
    "bookshelf": "open vertical bookshelf",
    "plant_pot": "ceramic plant pot (with a generic green plant)",
    "wall_art":  "framed wall print",
}

# Legacy style table — matches the prompt shape that produced the
# images_legacy_colors set for the 6 backed-up categories.
STYLES = {
    "var_a": ("modern", [
        "clean geometric lines",
        "neutral palette (off-white / pale grey) with a single accent colour",
        "contemporary minimal-but-warm aesthetic",
        "smooth surfaces",
    ]),
    "var_b": ("minimalist", [
        "near-monochromatic palette (white / pale beige / soft grey)",
        "extremely sparse silhouette",
        "no ornamentation or pattern",
        "Scandinavian-Japanese (Japandi) sensibility",
    ]),
    "var_c": ("vintage", [
        "1960-70s retro styling",
        "warm muted tones (mustard, olive, terracotta) and tactile patterns",
        "subtle ornamentation and well-worn patina",
        "mid-century feel",
    ]),
    "var_d": ("industrial", [
        "exposed metal hardware (raw steel, blackened iron) or unfinished wood",
        "utilitarian rectilinear forms",
        "neutral greys, browns and concrete tones",
        "loft / warehouse aesthetic",
    ]),
}

TIERS = {
    1: "Bold rendering — style markers are unmistakable; high colour and texture contrast.",
    2: "Moderate rendering — style is clearly visible but slightly toned down.",
    3: "Subtle rendering — style cues are present but understated; could almost pass as a generic version of this product.",
}


def build_prompt(cat_slug: str, var_key: str, tier: int) -> str:
    noun = CATEGORIES[cat_slug]
    style_slug, cues = STYLES[var_key]
    return (
        f"A photorealistic product photo of a single {noun} in {style_slug} style, "
        f"isolated on a pure white seamless studio background, soft even studio lighting, "
        f"centred composition, square 1:1 framing.\n"
        f"Style cues: {'; '.join(cues)}.\n"
        f"Grain: {TIERS[tier]}\n"
        f"Strict requirements: no people, no text, no watermarks, no logos, no other "
        f"products in frame, no shadows on the background floor. The product should "
        f"fill roughly 70% of the frame and be perfectly upright."
    )


def call_gemini(api_key: str, prompt: str, timeout: float = 90.0) -> bytes:
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode("utf-8")
    req = request.Request(
        ENDPOINT,
        data=body,
        headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("No inlineData in response: " + json.dumps(payload)[:600])


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("missing GEMINI_API_KEY")
    jobs: list[tuple[str, str, int]] = []
    for cat in CATEGORIES:
        for vk in STYLES:
            for tier in TIERS:
                jobs.append((cat, vk, tier))
    print(f"[regen-legacy] {len(jobs)} jobs")
    done = 0
    failed: list[tuple[str, str, int]] = []
    for i, (cat, vk, tier) in enumerate(jobs):
        path = PUBLIC_IMAGES / cat / f"{vk}_t{tier}.png"
        prompt = build_prompt(cat, vk, tier)
        tag = f"{cat}/{vk}/t{tier}"
        print(f"[regen-legacy] {i + 1:>2}/{len(jobs)}  {tag} …", flush=True, end=" ")
        t0 = time.time()
        try:
            img = call_gemini(api_key, prompt)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img)
            print(f"ok ({len(img) // 1024} KB, {time.time() - t0:.1f}s)")
            done += 1
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            print(f"FAIL http {e.code}: {body}")
            failed.append((cat, vk, tier))
            if e.code in (429, 500, 502, 503):
                time.sleep(5)
        except Exception as e:
            print(f"FAIL {type(e).__name__}: {e}")
            failed.append((cat, vk, tier))
    # One-shot retry of failures.
    if failed:
        print(f"\n[regen-legacy] retrying {len(failed)} failed jobs")
        for cat, vk, tier in failed:
            path = PUBLIC_IMAGES / cat / f"{vk}_t{tier}.png"
            prompt = build_prompt(cat, vk, tier)
            tag = f"{cat}/{vk}/t{tier}"
            print(f"[retry] {tag} …", flush=True, end=" ")
            try:
                img = call_gemini(api_key, prompt)
                path.write_bytes(img)
                print(f"ok ({len(img) // 1024} KB)")
                done += 1
            except Exception as e:
                print(f"FAIL again: {e}")
    print(f"\n[regen-legacy] done={done} of {len(jobs)}, still-failed={len(jobs) - done}")


if __name__ == "__main__":
    main()
