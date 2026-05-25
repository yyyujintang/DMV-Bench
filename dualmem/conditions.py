"""Cartesian condition grid: TaskCells × baseline systems.

Proposal_A target grid:
    10 categories × 4 variants × 3 tiers × 5 leakage × 3 mechanisms = ~1800 cells
    × 3 ceilings = ~5400 trials per memory system × N seeds
With 5+ memory systems and N>=10 seeds, the full grid is ~270K trials.

For pilot runs we expose a small_grid() that produces a representative
subset (~50-200 trials per ceiling per system) for end-to-end smoke
tests at minutes-scale cost.
"""

from __future__ import annotations

from itertools import product
from typing import List, Optional

from dualmem.inventory.spec import CATEGORIES, VARIANTS_PER_CATEGORY
from dualmem.types import TaskCell


def full_grid(
    categories: Optional[List[str]] = None,
    tiers: List[int] = (1, 2, 3),
    leakage_levels: List[int] = (0, 1, 2, 3, 4),
    mechanisms: List[str] = ("descriptive",),
    ceilings: List[str] = ("perception", "oracle_retrieval", "full_pipeline"),
    n_seeds: int = 10,
) -> List[TaskCell]:
    cat_ids = categories or [c["id"] for c in CATEGORIES]
    cells: List[TaskCell] = []
    for cat in cat_ids:
        for v_idx in range(VARIANTS_PER_CATEGORY):
            variant = f"var_{chr(ord('a') + v_idx)}"
            for tier, leak, mech, ceil, seed in product(tiers, leakage_levels, mechanisms, ceilings, range(n_seeds)):
                cells.append(TaskCell(
                    category=cat, variant=variant,
                    grain_tier=tier, leakage_level=leak,
                    mechanism=mech, ceiling=ceil, seed=seed,
                ))
    return cells


def small_grid(seed: int = 0) -> List[TaskCell]:
    """Pilot subset: 3 cats × 4 variants × 3 tiers × {0, 4} leakage × descriptive × 3 ceilings × 2 seeds.
    → 3 × 4 × 3 × 2 × 1 × 3 × 2 = 432 cells (144 trials per ceiling).
    """
    return full_grid(
        categories=["sofa", "lamp", "vase"],
        tiers=[1, 2, 3],
        leakage_levels=[0, 4],
        mechanisms=["descriptive"],
        ceilings=["perception", "oracle_retrieval", "full_pipeline"],
        n_seeds=2,
    )


def micro_grid() -> List[TaskCell]:
    """Smallest perception-only grid: 2 cats × 4 variants × 3 tiers × 1 leakage × 1 seed.
    24 perception calls only. ~30 s.
    """
    return full_grid(
        categories=["sofa", "vase"],
        tiers=[1, 2, 3],
        leakage_levels=[4],
        mechanisms=["descriptive"],
        ceilings=["perception"],
        n_seeds=1,
    )


def tiny_grid(category: str = "vase") -> List[TaskCell]:
    """Smallest all-ceilings grid: 1 cat × 4 variants × 3 tiers × 1 leakage × 1 seed × 3 ceilings.
    12 cells → 4 perception + 4 oracle×N_sys + 4 full×N_sys = ~36 API calls with 4 baselines.
    """
    return full_grid(
        categories=[category],
        tiers=[1, 2, 3],
        leakage_levels=[4],
        mechanisms=["descriptive"],
        ceilings=["perception", "oracle_retrieval", "full_pipeline"],
        n_seeds=1,
    )


def split_by_ceiling(cells: List[TaskCell]) -> dict:
    """Group cells by ceiling so we can dispatch to per-ceiling runners."""
    out = {"perception": [], "oracle_retrieval": [], "full_pipeline": []}
    for c in cells:
        out[c.ceiling].append(c)
    return out
