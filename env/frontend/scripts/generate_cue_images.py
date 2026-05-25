"""
Generate the W3 peripheral-cue catalogue (proposal_website.md §6).

20 cues across 5 types, each rendered as a small PNG with a transparent
background so it composites on top of the existing page chrome:

    carpet  (4)  — woven swatches for corner_bl position
    sticker (4)  — small graphic markers for banner position
    badge   (4)  — product badges for badge_tl position
    background (4) — subtle tile patterns
    lighting   (4) — soft colour washes overlaid on the page

Output: public/images/cues/<cue_key>.png

These cues are visual-only by contract — they MUST NEVER appear in any
text on the page. The CueOverlay React component renders them positioned
absolutely on top of normal chrome.

Usage:
  GEMINI_API_KEY=... python scripts/generate_cue_images.py
  python scripts/generate_cue_images.py --only carpet_checker
  python scripts/generate_cue_images.py --dry
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "public" / "images" / "cues"

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def cue(key: str, ctype: str, descriptor: str, default_position: str, salience: float, prompt_seed: str):
    return {
        "key": key,
        "type": ctype,
        "descriptor": descriptor,
        "default_position": default_position,
        "salience": salience,
        "prompt_seed": prompt_seed,
    }


# 20 cues — IDs are stable across runs; the seed is the prompt-specific text
# that distinguishes one cue from another within its type.
CUES = [
    # Carpet swatches — small textile squares
    cue("carpet_checker",  "carpet",  "woven black-and-cream checkerboard carpet",          "corner_bl", 0.45,
        "a small square swatch of checkerboard-pattern woven carpet, alternating black and cream, traditional weave"),
    cue("carpet_persian",  "carpet",  "ornate persian medallion carpet in burgundy and gold","corner_bl", 0.55,
        "a small square swatch of ornate persian-style carpet, deep burgundy field with a gold central medallion"),
    cue("carpet_heather",  "carpet",  "plain heather-blue tweed carpet swatch",             "corner_bl", 0.20,
        "a small square swatch of plain heather-blue tweed carpet, soft melange texture"),
    cue("carpet_grid",     "carpet",  "diamond-grid beige carpet swatch",                   "corner_bl", 0.30,
        "a small square swatch of beige carpet with a subtle diamond grid pattern"),

    # Stickers — graphic markers
    cue("sticker_red_dot", "sticker", "bold red round sticker",                              "banner",    0.65,
        "a single bright red circular sticker with a slight glossy highlight, looks like a peelable round sticker"),
    cue("sticker_blue_geo","sticker", "blue geometric polygonal sticker",                    "banner",    0.55,
        "a blue geometric polygonal sticker with sharp corners and slight 3D bevel"),
    cue("sticker_holo",    "sticker", "iridescent holographic round sticker",                "banner",    0.70,
        "a small round iridescent holographic sticker, rainbow shimmer"),
    cue("sticker_vintage", "sticker", "peeling vintage cream-and-brown rectangular label",   "banner",    0.30,
        "a small rectangular vintage paper label, slightly peeling, cream paper with brown distressed ink"),

    # Badges — product-corner labels
    cue("badge_sale",      "badge",   "red SALE rectangular tag",                            "badge_tl",  0.75,
        "a small red rectangular product tag with the word SALE in bold white block letters"),
    cue("badge_limited",   "badge",   "gold LIMITED rosette badge",                          "badge_tl",  0.65,
        "a small gold rosette badge with ruffled edges and the word LIMITED in elegant script in the centre"),
    cue("badge_new",       "badge",   "orange NEW starburst badge",                          "badge_tl",  0.70,
        "a small orange starburst badge with the word NEW in white sans-serif in the centre"),
    cue("badge_pick",      "badge",   "dark green EDITOR'S PICK ribbon",                     "badge_tl",  0.40,
        "a small dark forest-green diagonal ribbon banner with the phrase EDITOR'S PICK in slim gold letters"),

    # Background patterns — subtle tiles
    cue("bg_herringbone",  "background", "subtle grey herringbone pattern (tile)",           "background", 0.20,
        "a seamless tileable herringbone pattern in two close shades of light grey, very subtle contrast, repeating texture"),
    cue("bg_polka",        "background", "cream polka dots on white (tile)",                 "background", 0.25,
        "a seamless tileable polka-dot pattern, very small cream dots on a near-white background, repeating texture"),
    cue("bg_terrazzo",     "background", "speckled terrazzo pattern (tile)",                 "background", 0.30,
        "a seamless tileable terrazzo pattern, neutral creamy base with grey and warm-brown speckled chips, repeating"),
    cue("bg_palm",         "background", "faint palm leaf silhouettes (tile)",               "background", 0.20,
        "a seamless tileable pattern of faint sage-green palm leaf silhouettes on a near-white background, repeating"),

    # Lighting washes — colour overlays
    cue("light_sunset",    "lighting", "warm orange-pink sunset wash",                       "overlay",   0.30,
        "a soft radial gradient that fades from warm orange-pink at one corner to fully transparent at the opposite corner, no shapes, just a colour wash"),
    cue("light_blue",      "lighting", "cool blue diagonal wash from top-right",             "overlay",   0.30,
        "a soft diagonal gradient that fades from a cool sky-blue at the top-right to fully transparent at the bottom-left"),
    cue("light_golden",    "lighting", "warm amber golden-hour glow",                        "overlay",   0.35,
        "a soft radial glow of warm amber light centred slightly off-frame, fading to transparent at the edges, no shapes"),
    cue("light_neon",      "lighting", "vibrant pink-purple corner glow",                    "overlay",   0.50,
        "a soft corner glow of vibrant neon pink-to-purple from one corner, fading quickly to transparent"),
]


def build_prompt(c: dict) -> str:
    common = (
        "Square PNG with a fully transparent background (alpha channel). "
        "No people, no text labels except where explicitly specified in the description, "
        "no logos, no watermarks. The element must be cleanly cut out — only the "
        "element on transparent background, no surrounding frame, no rectangular crop. "
        "Centre-composed, the element fills roughly 80% of the canvas."
    )
    type_specific = {
        "carpet":     "Render as a flat overhead view of a small carpet swatch, square, with the woven texture clearly visible.",
        "sticker":    "Render as a small sticker shape (no background card behind it), as if it were applied to a surface.",
        "badge":      "Render as a small product badge shape only (no surrounding card or product behind it).",
        "background": "Render the pattern as a seamless tileable texture, square 1:1, suitable for repetition. The pattern should be subtle (low contrast).",
        "lighting":   "Render the gradient on a near-transparent canvas; the centre of the gradient is the most opaque region, fading to fully transparent at the edges.",
    }[c["type"]]
    return f"{c['prompt_seed']}.\n{type_specific}\n{common}"


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


def out_path(key: str) -> Path:
    return OUT_DIR / f"{key}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Generate one cue by key (e.g. carpet_checker)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry:
        sys.exit("missing GEMINI_API_KEY in env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [c for c in CUES if (not args.only or c["key"] == args.only)]
    if args.only and not targets:
        sys.exit(f"no cue with key {args.only!r}")

    done, skipped, failed = 0, 0, 0
    for i, c in enumerate(targets):
        path = out_path(c["key"])
        prompt = build_prompt(c)
        if args.dry:
            print(f"\n=== {c['key']} ({c['type']}) → {path.relative_to(ROOT)} ===")
            print(prompt)
            continue
        if not args.force and path.exists() and path.stat().st_size > 1024:
            print(f"[cue] skip  {c['key']} (already present)")
            skipped += 1
            continue
        print(f"[cue] {i + 1:>2}/{len(targets)} {c['key']} …", flush=True, end=" ")
        t0 = time.time()
        try:
            img = call_gemini(api_key, prompt)
            path.write_bytes(img)
            elapsed = time.time() - t0
            print(f"ok ({len(img) // 1024} KB, {elapsed:.1f}s)")
            done += 1
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:400]
            print(f"FAIL http {e.code}: {body}")
            failed += 1
        except Exception as e:
            print(f"FAIL {type(e).__name__}: {e}")
            failed += 1

    print(f"\n[cue] done={done} skipped={skipped} failed={failed}")
    if not args.dry:
        # Emit a manifest the seed script can consume.
        manifest = [
            {"cueKey": c["key"], "cueType": c["type"], "visualDescriptor": c["descriptor"],
             "assetUrl": f"/images/cues/{c['key']}.png", "defaultPosition": c["default_position"],
             "salienceScore": c["salience"]}
            for c in CUES
        ]
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[cue] manifest → {(OUT_DIR / 'manifest.json').relative_to(ROOT)}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
