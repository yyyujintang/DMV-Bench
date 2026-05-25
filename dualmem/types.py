"""Core schemas for VisMem-Diag.

TaskCell defines ONE experimental unit on the 3-axis grid.
Trial is the runtime data the agent sees (anchor + 4 candidates).
CellResult is the single-row output of running one trial.

Design choice: TaskCell is fully self-describing (carries category /
variant / tier / leakage / mechanism / ceiling / seed) so any result row
can be joined back to its grid coordinates without auxiliary tables.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class GrainTier(Enum):
    TIER1 = 1   # coarse: e.g. red vs blue
    TIER2 = 2   # medium: e.g. red vs orange
    TIER3 = 3   # fine: e.g. charcoal vs slate


class LeakageLevel(Enum):
    L0 = 0   # masked: no descriptive text, generic title only
    L1 = 1   # coarse class: "furniture", "lamp"
    L2 = 2   # material/function: "upholstered seating"
    L3 = 3   # specific features: "gray loveseat"
    L4 = 4   # full marketing copy: realistic e-commerce description


class Mechanism(Enum):
    DESCRIPTIVE = "descriptive"       # fine-grained attribute recall
    ANALOGUE = "analogue"             # cross-category style match
    TIP_OF_TONGUE = "tip_of_tongue"   # long-filler recall (50-100 steps)


class Ceiling(Enum):
    PERCEPTION = "perception"            # VLM 4AFC, no memory mediation
    ORACLE_RETRIEVAL = "oracle_retrieval"  # anchor injected directly, skip retrieval
    FULL_PIPELINE = "full_pipeline"      # standard encode→gate→retrieve→inject


@dataclass(frozen=True)
class TaskCell:
    """One cell on the (cat × variant × tier × leakage × mech × ceiling × seed) grid."""
    category: str
    variant: str           # one of the 4 variants for this category
    grain_tier: int        # 1..3
    leakage_level: int     # 0..4
    mechanism: str         # Mechanism value
    ceiling: str           # Ceiling value
    seed: int

    @property
    def cell_id(self) -> str:
        """Deterministic hash for storage / lookup."""
        key = f"{self.category}|{self.variant}|t{self.grain_tier}|l{self.leakage_level}|{self.mechanism}|{self.ceiling}|s{self.seed}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def asdict(self) -> dict:
        return {
            "category": self.category,
            "variant": self.variant,
            "grain_tier": self.grain_tier,
            "leakage_level": self.leakage_level,
            "mechanism": self.mechanism,
            "ceiling": self.ceiling,
            "seed": self.seed,
            "cell_id": self.cell_id,
        }


@dataclass
class Trial:
    """One runtime task instance fed to the agent.

    - anchor_path: the variant the agent must remember
    - candidate_paths: 4 candidates in display order (1 correct, 3 distractors)
    - correct_index: which slot holds the correct variant
    - encode_text: description shown at encode time (leakage-controlled)
    - recall_text: description shown at recall time (leakage-controlled, often same)
    - filler_steps: number of intermediate unrelated steps (for tip_of_tongue mechanism)
    """
    cell: TaskCell
    anchor_path: str
    anchor_slug: str
    candidate_paths: List[str]
    candidate_slugs: List[str]
    correct_index: int
    encode_text: str
    recall_text: str
    filler_steps: int = 0
    distractors_info: List[dict] = field(default_factory=list)


@dataclass
class CellResult:
    """One row in the results table."""
    cell: TaskCell
    system: str               # baseline system name (e.g. "TextOnly", "CoMEM")
    chosen_index: int         # -1 = parse failure
    correct: bool
    latency_ms: int
    raw_response: str
    trial_id: str             # hash of (cell_id, system) for join keys
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def asdict(self) -> dict:
        d = self.cell.asdict()
        d.update({
            "system": self.system,
            "chosen_index": self.chosen_index,
            "correct": int(self.correct),
            "latency_ms": self.latency_ms,
            "trial_id": self.trial_id,
            "error": self.error or "",
        })
        for k, v in self.extra.items():
            d[f"x_{k}"] = v
        return d
