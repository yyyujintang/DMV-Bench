"""Baseline registry: name → factory mapping.

In-house baselines (deterministic, no external deps):
    NoMemory / TextOnly / Caption / DualChannel
                       (DualChannel = our DualMem)

External 2025-2026 multimodal-agent-memory baselines — all
**model-agnostic** so the same baseline runs across every backbone in
our 1-open + 3-closed grid:

    WorldMM   — Yeo et al., CVPR 2026 Highlight, arXiv:2512.02425
    M2A       — Feng et al., arXiv:2602.07624 (Feb 2026)
    MMA       — Lu et al.,   arXiv:2602.16493 (Feb 2026)

See `dualmem/baselines/external/` and `doc/baselines_external_adapters.md`.
"""

from __future__ import annotations

from typing import Callable, Dict

from dualmem.baselines.text_only import TextOnly
from dualmem.baselines.caption import Caption
from dualmem.baselines.dual_channel import DualChannel

# External adapters: imports are cheap; external repos and weights are
# pulled lazily on first encode/retrieve call, not on import.
from dualmem.baselines.external.worldmm import WorldMMAdapter
from dualmem.baselines.external.m2a import M2AAdapter
from dualmem.baselines.external.mma import MMAAdapter


BASELINES: Dict[str, Callable[[], object]] = {
    # In-house
    "TextOnly":    TextOnly,
    "Caption":     Caption,
    "DualChannel": DualChannel,
    # External (model-agnostic, see external/*.py docstrings)
    "WorldMM":     WorldMMAdapter,
    "M2A":         M2AAdapter,
    "MMA":         MMAAdapter,
}


def make_baseline(name: str):
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name!r}. Known: {list(BASELINES)}")
    return BASELINES[name]()
