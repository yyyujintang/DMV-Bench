"""Shared helpers for external-baseline adapters."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional


# Where setup_external_baselines.sh drops the cloned repos.
EXTERNAL_REPO_ROOT = Path(
    os.environ.get(
        "DMV_EXTERNAL_REPO_ROOT",
        str(Path(__file__).resolve().parents[3] / "external" / "repos"),
    )
)


def repo_path(name: str) -> Path:
    """Path where `setup_external_baselines.sh` clones repo `<name>`."""
    return EXTERNAL_REPO_ROOT / name


def ensure_repo_on_path(name: str) -> Path:
    """Add `external/repos/<name>` to sys.path; raise if not cloned."""
    p = repo_path(name)
    if not p.exists():
        raise RuntimeError(
            f"External repo {name!r} not found at {p}. "
            f"Run `scripts/setup_external_baselines.sh` first."
        )
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return p


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def token_overlap(a: str, b: str) -> int:
    return len(tokens(a) & tokens(b))


class ExternalBaselineUnavailable(RuntimeError):
    """Raised when an adapter is exercised but its external repo / weights aren't ready."""
