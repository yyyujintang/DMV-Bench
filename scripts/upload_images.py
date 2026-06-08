#!/usr/bin/env python
"""Publish the DMV-Bench catalogue (`with_cue` set) to a Hugging Face dataset.

Companion to scripts/download_images.py. Uploads the ~1,000-variant
`with_cue` image tree (<category>/<style>/NN.png) to a Hub *dataset* so a
fresh checkout can fetch it. The dataset stores the category tree at its root,
which is the layout download_images.py expects.

Usage:
    export DMVBENCH_IMAGES_REPO="<hf-namespace>/dmvbench-images"
    export HF_TOKEN="hf_..."                 # a write token
    python scripts/upload_images.py          # uploads data/.../images/with_cue
    python scripts/upload_images.py --src /path/to/with_cue --create --private
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "<hf-namespace>/dmvbench-images"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "data" / "vismem_diag_v2" / "images" / "with_cue"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=os.environ.get("DMVBENCH_IMAGES_REPO",
                                                       DEFAULT_REPO_ID))
    p.add_argument("--src", default=str(DEFAULT_SRC),
                   help="local with_cue folder (<category>/<style>/NN.png).")
    p.add_argument("--create", action="store_true",
                   help="create the dataset repo if it does not exist.")
    p.add_argument("--private", action="store_true",
                   help="create as private (only with --create).")
    args = p.parse_args()

    if "<" in args.repo_id or args.repo_id == DEFAULT_REPO_ID:
        print("error: set --repo-id or DMVBENCH_IMAGES_REPO to the target dataset.",
              file=sys.stderr)
        return 2
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("error: set HF_TOKEN (a write token) in the environment.", file=sys.stderr)
        return 2
    src = Path(args.src)
    n = sum(1 for _ in src.rglob("*.png")) if src.is_dir() else 0
    if n == 0:
        print(f"error: no .png images under {src}", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if args.create:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                        private=args.private, exist_ok=True)
        print(f"[upload] ensured dataset repo {args.repo_id} (private={args.private})")

    print(f"[upload] pushing {n} images from {src} -> {args.repo_id}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(src),
        print_report=True,
    )
    print(f"[upload] done: {n} images -> {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
