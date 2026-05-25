"""
Generate 120 catalog images, one per variant, using the W7r2
material × style × color taxonomy.

Source of truth: `env/scripts/imaging_specs.json`, produced by
`generate_pricing_naming.py`. Each urlHash entry contains:
  - categorySlug, noun, material, style, color, imagePath

Output: replaces files at `public/<imagePath>` (e.g. `public/images/sofa/var_a_t1.png`).

Usage:
  GEMINI_API_KEY=… python scripts/generate_style_images.py
  GEMINI_API_KEY=… python scripts/generate_style_images.py --only <urlHash>
  python scripts/generate_style_images.py --dry
  python scripts/generate_style_images.py --force
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib import request, error

FRONTEND_ROOT = Path(__file__).resolve().parent.parent
ENV_ROOT = FRONTEND_ROOT.parent
IMAGING_SPECS = ENV_ROOT / "scripts" / "imaging_specs.json"
PUBLIC = FRONTEND_ROOT / "public"
LEGACY_BACKUP = FRONTEND_ROOT / "public" / "images_w7r1_backup"

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


# Style → silhouette / construction features that DEFINE the style independently
# of colour. Colour is a secondary accent — the per-variant spec['color'] still
# appears in the prompt but the style cues come from form, not hue. This is the
# W7r3 "Option C" pivot — see proposal_tasks.md note on style/color decoupling.
FORM_FEATURES: dict[str, str] = {
    "modern": (
        "geometric and rectilinear silhouette, clean straight lines, flat planes, "
        "slim metal-or-natural-wood accents, no ornamentation, no curves on the body"
    ),
    "minimalist": (
        "extremely sparse single-continuous-surface form, monolithic block-like body, "
        "no visible joinery, no ornament, hidden hardware, the simplest possible profile"
    ),
    "vintage": (
        "curved silhouette with tufting, piping or carved/turned details, rolled arms "
        "or scalloped edges, decorative trim, period mid-century-or-earlier proportions"
    ),
    "industrial": (
        "exposed bolts, rivets or weld seams, raw metal framework, utilitarian "
        "rectangular bracing, visible joinery, rough finished surfaces, loft aesthetic"
    ),
}


def build_prompt(spec: dict) -> str:
    style = spec["style"]
    form = FORM_FEATURES[style]
    return (
        f"A photorealistic e-commerce product photo of a single {spec['noun']}, "
        f"isolated on a pure white seamless studio background, soft even studio lighting, "
        f"centred composition, square 1:1 framing. "
        f"FORM (this is the dominant style cue and must be unambiguous): {form}. "
        f"MATERIAL (visible from surface texture and finish): {spec['material']}. "
        f"ACCENT COLOUR (secondary, on the body): {spec['color']}. "
        f"The form features must be readable at a glance regardless of colour — "
        f"a viewer should classify the style from silhouette alone. "
        f"Strict: no people, no text, no watermarks, no logos, no other products in "
        f"frame, no shadows on the background. The product fills ~70% of the frame "
        f"and is perfectly upright."
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


def resolve_target(spec: dict) -> Path:
    rel = spec["imagePath"].lstrip("/")
    return PUBLIC / rel


def backup_once() -> None:
    if LEGACY_BACKUP.exists():
        return
    src = PUBLIC / "images"
    if not src.exists():
        return
    print(f"[gen] backing up current images → {LEGACY_BACKUP}", flush=True)
    shutil.copytree(src, LEGACY_BACKUP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Single urlHash to regenerate")
    ap.add_argument("--dry", action="store_true", help="Print prompts, don't call API")
    ap.add_argument("--force", action="store_true", help="Regenerate even if file exists")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N successful generations")
    ap.add_argument("--start", type=int, default=0, help="Start at index N (for resuming)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry:
        sys.exit("missing GEMINI_API_KEY in env")

    specs = json.loads(IMAGING_SPECS.read_text())
    items = list(specs.items())
    if args.only:
        items = [(args.only, specs[args.only])]
    elif args.start:
        items = items[args.start:]

    if not args.dry and not args.only and not args.force:
        backup_once()

    done, skipped, failed = 0, 0, 0
    for i, (urlhash, spec) in enumerate(items):
        target = resolve_target(spec)
        tag = f"{urlhash} ({spec['categorySlug']}/{spec['style']}/{spec['color']})"
        prompt = build_prompt(spec)
        if args.dry:
            print(f"\n=== {tag} → {target.relative_to(FRONTEND_ROOT)} ===")
            print(prompt)
            continue
        if args.limit and done >= args.limit:
            print(f"[gen] reached --limit={args.limit}, stopping")
            break
        if not args.force and target.exists() and target.stat().st_size > 1024:
            # Check if backup of pre-existing image exists; if not we're
            # on a fresh run and should regenerate.  The backup is the
            # signal that "current images = W7r2 generation".
            if LEGACY_BACKUP.exists():
                skipped += 1
                continue
        print(f"[gen] {i + 1:>3}/{len(items)}  {tag} …", flush=True, end=" ")
        t0 = time.time()
        try:
            img = call_gemini(api_key, prompt)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(img)
            elapsed = time.time() - t0
            print(f"ok ({len(img) // 1024} KB, {elapsed:.1f}s)")
            done += 1
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:600]
            print(f"FAIL http {e.code}: {body}")
            failed += 1
            # Backoff on 429/5xx, otherwise continue.
            if e.code in (429, 500, 502, 503):
                time.sleep(5)
        except Exception as e:
            print(f"FAIL {type(e).__name__}: {e}")
            failed += 1

    print(f"\n[gen] done={done} skipped={skipped} failed={failed}")
    if failed > 0 and done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
