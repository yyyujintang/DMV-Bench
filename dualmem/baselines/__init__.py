"""Adapters for the external multimodal-agent-memory systems we compare against.

Each is a faithful-as-feasible re-implementation of a published system inside
the DMV-Bench harness; see `external/` for what was kept and what was dropped
per adapter.

    WorldMM  -- tri-modular memory (episodic / semantic / visual) with
                adaptive iterative retrieval
    M2A      -- dual-layer memory: a raw store plus a semantic-abstraction
                store, agent-routed
    MMA      -- reliability-aware retrieval, weighting items by source
                credibility, temporal decay and conflict-aware consensus

They are wrapped into the shared encode/retrieve/inject interface by
`dualmem/systems/external_wrap.py` and reached through
`dualmem/systems/registry.py`.
"""

from dualmem.baselines.external import WorldMMAdapter, M2AAdapter, MMAAdapter

__all__ = ["WorldMMAdapter", "M2AAdapter", "MMAAdapter"]
