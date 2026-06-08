#!/usr/bin/env python
"""Fetch the DMV-Bench catalogue images from the Hugging Face Hub.

The ~1,000-variant home-furnishing catalogue (~1.5 GB) is too large to ship in
git, so it is hosted as a Hub *dataset*. Design v6: every storefront product
carries a unique baked-in incidental cue, so the runner and the storefront use
the same `with_cue` image set — that is the only set published on the Hub:

    <category>/<style>/NN.png       product photo with its baked-in cue

This script downloads it and lays it out where the code expects:

    data/vismem_diag_v2/images/with_cue/     <- read by the runner
    env/frontend/public/images_v2/           <- served by the storefront

Both destinations hold the same images; pass --link-frontend to symlink the
second to the first instead of copying (saves ~1.5 GB).

Usage:
    export DMVBENCH_IMAGES_REPO="<hf-namespace>/dmvbench-images"
    python scripts/download_images.py
    python scripts/download_images.py --link-frontend   # symlink, don't copy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Public Hugging Face dataset holding the catalogue (`with_cue` set).
# Override with --repo-id or the DMVBENCH_IMAGES_REPO env var.
DEFAULT_REPO_ID = "yyyujintang/DMV-Bench-Images"

REPO_ROOT = Path(__file__).resolve().parents[1]
WITH_CUE_DIR = REPO_ROOT / "data" / "vismem_diag_v2" / "images" / "with_cue"
FRONTEND_IMAGES_V2 = REPO_ROOT / "env" / "frontend" / "public" / "images_v2"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=os.environ.get("DMVBENCH_IMAGES_REPO",
                                                       DEFAULT_REPO_ID),
                   help="Hugging Face dataset repo id holding the with_cue catalogue.")
    p.add_argument("--revision", default=None, help="Optional dataset revision / tag.")
    p.add_argument("--link-frontend", action="store_true",
                   help="symlink env/frontend/public/images_v2 to the with_cue set "
                        "instead of copying (saves ~1.5 GB).")
    args = p.parse_args()

    if "<" in args.repo_id:
        print("error: no image dataset repo id configured.\n"
              "  set --repo-id or the DMVBENCH_IMAGES_REPO environment variable to\n"
              "  the Hugging Face dataset that hosts the catalogue images.",
              file=sys.stderr)
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub is required: pip install huggingface_hub",
              file=sys.stderr)
        return 2

    WITH_CUE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[images] downloading {args.repo_id} -> {WITH_CUE_DIR}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(WITH_CUE_DIR),
        allow_patterns=["*/*/*.png", "*/*.png"],
    )

    # snapshot_download leaves a .cache/huggingface/ bookkeeping dir inside the
    # target; drop it so the runner/storefront image roots hold images only.
    import shutil
    cache_dir = WITH_CUE_DIR / ".cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)

    n = sum(1 for _ in WITH_CUE_DIR.rglob("*.png"))
    if n == 0:
        print(f"error: no .png images found under {WITH_CUE_DIR} after download — "
              f"check the dataset layout (expected <category>/<style>/NN.png).",
              file=sys.stderr)
        return 1

    # Mirror to the storefront-served root (identical content).
    if FRONTEND_IMAGES_V2.exists() or FRONTEND_IMAGES_V2.is_symlink():
        print(f"  [skip] {FRONTEND_IMAGES_V2} already exists")
    else:
        FRONTEND_IMAGES_V2.parent.mkdir(parents=True, exist_ok=True)
        if args.link_frontend:
            FRONTEND_IMAGES_V2.symlink_to(WITH_CUE_DIR)
            print(f"  [link] {FRONTEND_IMAGES_V2} -> {WITH_CUE_DIR}")
        else:
            import shutil
            print(f"  [copy] {WITH_CUE_DIR} -> {FRONTEND_IMAGES_V2}  (this can take a minute)")
            shutil.copytree(WITH_CUE_DIR, FRONTEND_IMAGES_V2)

    print(f"[images] done: {n} with_cue images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
