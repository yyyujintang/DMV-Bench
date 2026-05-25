"""AgentTask — one (anchor, filler, recall) episode for the Playwright agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class AgentTask:
    task_id: str
    mechanism: str                 # "same_instance" | "cross_category" | "long_horizon"
    category: str                  # "chairs"
    grain_tier: int                # 1, 2, 3
    leakage_level: int             # 0..4
    seed: int

    # Phase 1: anchor
    anchor_url: str                # "/product/37de2f75"
    anchor_slug: str               # "chair0" — for OracleRetriever
    anchor_image_path: str         # "$DATA_ROOT/images/chair/var_a_t1.png"

    # Phase 2: filler URLs (visited in order). Pre-seeded.
    filler_urls: List[str] = field(default_factory=list)

    # Phase 3: recall
    recall_instruction: str = ""   # natural-language task
    recall_collection_url: str = "" # /category/chairs?tier=3 where candidates live
    expected_url_pattern: str = "" # regex anchor_url must match (e.g. r"^/product/37de2f75")

    # for analysis only
    candidate_urls: List[str] = field(default_factory=list)   # 4 product URLs at collection
    correct_index: int = -1                                    # which candidate is the anchor

    def asdict(self) -> dict:
        return asdict(self)
