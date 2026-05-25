"""Snapshot-before-overwrite + rollback for generated images.

Every time the driver is about to overwrite a PNG, it first copies the
existing file into `images_history/<tag>/<relpath>`. The tag identifies
the run that PRODUCED the file being replaced — typically the
prompt_hash of the latest manifest record for that urlHash, or the
literal "pre-history" when no prior manifest record exists.

This recovers the bug the user hit on 2026-05-15, where 42 PNGs were
overwritten in place by a prompt rewrite the user later wanted to
revert. With this module wired in, any future overwrite is reversible
by walking the history tree.

CLI usage:

  python3 -m pipeline.history list             # snapshots + counts
  python3 -m pipeline.history rollback <hash>  # restore live ← snapshot
  python3 -m pipeline.history diff <hash>      # files differing live vs snapshot
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE_ROOT = REPO / "VisMem-Diag" / "data" / "vismem_diag" / "images"
HISTORY_ROOT = REPO / "VisMem-Diag" / "data" / "vismem_diag" / "images_history"

PRE_HISTORY_TAG = "pre-history"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_before_write(target: Path, tag: str | None) -> Path | None:
    """If `target` already exists, copy it into history under `tag`.

    Returns the snapshot path on success, None when target absent.
    The tag should uniquely identify the run that produced the file
    currently at `target` — convention: `<backend_id>__<prompt_hash>`.
    Pass None and we fall back to PRE_HISTORY_TAG.

    Idempotent: if the same source content is already snapshotted at
    the same path with the same sha256, skip the copy.
    """
    if not target.exists():
        return None
    # The driver writes via the symlinked path
    # (VisMem-Diag/env/frontend/public/images/...). Resolve to the
    # real filesystem location (VisMem-Diag/data/vismem_diag/images/...)
    # so `relative_to(LIVE_ROOT)` succeeds. Without this resolve, the
    # symlinked path doesn't sit under LIVE_ROOT and snapshot is
    # silently skipped — that's the 2026-05-15 Qwen-smoke bug where
    # var_b_t1.png got overwritten without a history copy.
    resolved = target.resolve()
    try:
        rel = resolved.relative_to(LIVE_ROOT.resolve())
    except ValueError:
        # target lives outside LIVE_ROOT — bail rather than guess
        return None
    snap = HISTORY_ROOT / (tag or PRE_HISTORY_TAG) / rel
    if snap.exists() and _sha256(snap) == _sha256(target):
        return snap
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, snap)
    return snap


def list_snapshots() -> dict[str, int]:
    """Map snapshot tag → number of files under it."""
    out: dict[str, int] = {}
    if not HISTORY_ROOT.exists():
        return out
    for tag_dir in sorted(p for p in HISTORY_ROOT.iterdir() if p.is_dir()):
        count = sum(1 for _ in tag_dir.rglob("*") if _.is_file())
        out[tag_dir.name] = count
    return out


def rollback(tag: str, dry_run: bool = False) -> list[Path]:
    """Copy every file under images_history/<tag>/ back to LIVE_ROOT.

    Before any restore, the current live file is itself snapshotted
    under a tag-of-the-tag (`<tag>__pre-rollback`) so the rollback is
    itself reversible.

    Returns the list of live paths touched (or that would be touched
    in dry_run mode).
    """
    src_root = HISTORY_ROOT / tag
    if not src_root.exists():
        raise FileNotFoundError(f"no snapshot at {src_root}")
    touched: list[Path] = []
    pre_rollback_tag = f"{tag}__pre-rollback"
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        live = LIVE_ROOT / rel
        touched.append(live)
        if dry_run:
            continue
        if live.exists():
            # Snapshot the file we're about to overwrite, so the
            # rollback itself is reversible.
            backup = HISTORY_ROOT / pre_rollback_tag / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(live, backup)
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, live)
    return touched


def diff(tag: str) -> dict[str, str]:
    """For each file in the snapshot, report whether the live file
    matches ('same'), differs ('changed'), or is missing ('absent')."""
    src_root = HISTORY_ROOT / tag
    if not src_root.exists():
        raise FileNotFoundError(f"no snapshot at {src_root}")
    out: dict[str, str] = {}
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        live = LIVE_ROOT / rel
        if not live.exists():
            out[str(rel)] = "absent"
        elif _sha256(live) == _sha256(src):
            out[str(rel)] = "same"
        else:
            out[str(rel)] = "changed"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="pipeline.history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list snapshot tags + file counts")
    p_rb = sub.add_parser("rollback", help="restore a snapshot in place")
    p_rb.add_argument("tag")
    p_rb.add_argument("--dry", action="store_true")
    p_df = sub.add_parser("diff", help="show which files differ live vs snapshot")
    p_df.add_argument("tag")
    args = ap.parse_args()
    if args.cmd == "list":
        snaps = list_snapshots()
        if not snaps:
            print("no snapshots yet")
            return 0
        for tag, n in snaps.items():
            print(f"{tag}\t{n} files")
        return 0
    if args.cmd == "rollback":
        touched = rollback(args.tag, dry_run=args.dry)
        verb = "would restore" if args.dry else "restored"
        print(f"{verb} {len(touched)} files from {args.tag}")
        return 0
    if args.cmd == "diff":
        d = diff(args.tag)
        for path, status in sorted(d.items()):
            print(f"{status}\t{path}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
