#!/usr/bin/env python
"""Render the incidental cue onto every base catalogue photograph.

Stage 3 of catalogue construction, after `pipeline/generate.py` (base photos)
and `scripts/generate_cue_edit_prompts.py` (cue assignment + prompts). Reads
`prompts/cue_edit_manifest.json` and, for each of the 1,000 products, calls the
Gemini image-edit backend with (base photo, edit prompt), writing the result to
`data/vismem_diag_v2/images/with_cue/<cat>/<style>/<idx>.png` -- the image set
both the runner and the storefront serve.

Resumable: an output that already exists is skipped unless --force. Rows whose
base photo is not yet on disk are skipped, so this can run alongside stage 1.

Most users should not need this -- `scripts/download_images.py` fetches the
released catalogue. Rebuild only if you want a fresh one.

    export GEMINI_API_KEY=...
    python scripts/run_cue_edits.py --limit 4      # smoke
    python scripts/run_cue_edits.py                # full (~1000 edits)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.backends.base import GenerateRequest          # noqa: E402
from pipeline.backends.gemini import GeminiBackend          # noqa: E402

V2_ROOT = REPO_ROOT / "data" / "vismem_diag_v2"
EDIT_MANIFEST = V2_ROOT / "prompts" / "cue_edit_manifest.json"
PROGRESS_LOG = V2_ROOT / "prompts" / "cue_edit_progress.json"


def _save_png(raw: bytes, out_path: Path, size: int) -> None:
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N rows (smoke test)")
    p.add_argument("--size", type=int, default=1024,
                   help="output edge length in pixels")
    p.add_argument("--force", action="store_true",
                   help="re-edit rows whose output already exists")
    args = p.parse_args()

    if not EDIT_MANIFEST.exists():
        print(f"missing {EDIT_MANIFEST} — run "
              f"scripts/generate_cue_edit_prompts.py first", file=sys.stderr)
        return 2

    rows = json.loads(EDIT_MANIFEST.read_text())["rows"]
    if args.limit is not None:
        rows = rows[: args.limit]

    backend = GeminiBackend()
    print(f"[cue-edit] {len(rows)} rows, size={args.size}, "
          f"backend={backend.backend_id}", flush=True)

    n_done = n_skip = n_fail = n_no_base = 0
    progress: list[dict] = []
    t_start = time.time()

    for i, row in enumerate(rows, start=1):
        base_path = REPO_ROOT / row["base_image_path"]
        out_path = REPO_ROOT / row["output_image_path"]
        cue = row["cue"]
        tag = f"{row['cat']}/{row['style']}/{row['prod_idx']:02d}"

        if out_path.exists() and not args.force:
            n_skip += 1
            continue
        if not base_path.exists():
            n_no_base += 1
            print(f"  [{i:4d}/{len(rows)}] no base photo yet  {tag}", flush=True)
            continue

        prompt = (REPO_ROOT / row["edit_prompt_path"]).read_text().strip()
        t0 = time.time()
        try:
            raw = backend.generate(GenerateRequest(
                prompt=prompt, source_image=base_path.read_bytes()))
            _save_png(raw, out_path, args.size)
            n_done += 1
            took = time.time() - t0
            eta = (time.time() - t_start) / n_done * (len(rows) - i - n_skip)
            print(f"  [{i:4d}/{len(rows)}] {tag}  +{cue['color']} {cue['object']}"
                  f"  {took:.1f}s  eta={eta / 60:.0f}min", flush=True)
            progress.append({"i": i, "status": "ok", "tag": tag,
                             "cue_id": f"{row['cat']}::{cue['object']}::{cue['color']}",
                             "elapsed_s": round(took, 2)})
        except Exception as e:
            n_fail += 1
            print(f"  [{i:4d}/{len(rows)}] FAILED {tag}  "
                  f"{type(e).__name__}: {e}", flush=True)
            progress.append({"i": i, "status": "fail", "tag": tag,
                             "error": f"{type(e).__name__}: {e}"})

        if i % 10 == 0:
            PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS_LOG.write_text(json.dumps(progress, indent=2))

    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text(json.dumps(progress, indent=2))
    print(f"[cue-edit] done. edited={n_done} skipped={n_skip} "
          f"no_base={n_no_base} failed={n_fail}  "
          f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
