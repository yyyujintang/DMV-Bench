"""External-paper baselines for DMV-Bench.

Final lineup is **three model-agnostic** multimodal-agent-memory
systems published 2025-2026 with public code:

| Adapter         | Paper                                                    | Repo                                |
|-----------------|----------------------------------------------------------|-------------------------------------|
| WorldMMAdapter  | WorldMM (CVPR 2026 Highlight, arXiv:2512.02425)          | github.com/wgcyeo/WorldMM           |
| M2AAdapter      | M2A: Multimodal Memory Agent with Dual-Layer Hybrid Memory (arXiv:2602.07624) | github.com/Little-Fridge/M2A |
| MMAAdapter      | MMA: Multimodal Memory Agent (arXiv:2602.16493)          | github.com/AIGeeksGroup/MMA         |

Why these three (vs. M3-Agent / CoMEM-Agent which we initially considered):
**model-agnosticism**. All three can be driven entirely through an API-
level VLM call, so the same baseline runs across our 1-open + 3-closed
backbone grid (Qwen2.5-VL-7B / Gemini 2.5 Pro / GPT-4o / Claude 3.7
Sonnet). M3-Agent (fine-tuned Qwen-Omni heads) and CoMEM (LoRA Q-Former
that injects continuous tokens into the VLM's input-embedding layer)
are model-locked by construction and were dropped to avoid surrogate
baselines in the main table.

See `doc/baselines_external_adapters.md` for adaptation notes per system.

External repos are NOT vendored in — clone them via
`scripts/setup_external_baselines.sh` into `external/repos/<name>/`.
Each adapter degrades gracefully to a 'spirit-of' local implementation
if the upstream repo isn't on disk.
"""

from dualmem.baselines.external.worldmm import WorldMMAdapter
from dualmem.baselines.external.m2a import M2AAdapter
from dualmem.baselines.external.mma import MMAAdapter

__all__ = ["WorldMMAdapter", "M2AAdapter", "MMAAdapter"]
