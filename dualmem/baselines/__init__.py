"""Memory baselines for DMV-Bench.

In-house baselines:
    TextOnly    — stores leakage text only
    Caption     — stores VLM-generated caption only
    DualChannel — caption + image (a.k.a. DualMem in the paper)

External, model-agnostic 2025-2026 multimodal-agent-memory baselines
(see `baselines/external/` and `doc/baselines_external_adapters.md`):

    WorldMM     — tri-modular memory + adaptive routing (CVPR 2026 Highlight)
    M2A         — dual-layer (raw + semantic) hybrid memory (Feb 2026)
    MMA         — reliability-aware retrieval (Feb 2026)

Every baseline exposes:
    name: str
    encode(trial)
    retrieve(query_text) -> Trial | None
    oracle_inject(trial) -> {"text": str, "images": [paths]}
    reset()
"""

from dualmem.baselines.text_only import TextOnly
from dualmem.baselines.caption import Caption
from dualmem.baselines.dual_channel import DualChannel
from dualmem.baselines.external import WorldMMAdapter, M2AAdapter, MMAAdapter
from dualmem.baselines.registry import BASELINES, make_baseline

__all__ = [
    "TextOnly", "Caption", "DualChannel",
    "WorldMMAdapter", "M2AAdapter", "MMAAdapter",
    "BASELINES", "make_baseline",
]
