"""Multisession composer — glues 3..5 sub-task instances into one MultiSessionTask.

For each composed task, the composer:
  1. Picks a variant (short_3 / short_4 / short_5 / long_3 / long_4 / long_5
     / homogeneous_*).
  2. Picks sub-task ids from the validated pool for each session slot.
  3. Wires `memory_contract.retrieves_from_prior` on session N to point at
     anchors that session N-1's `memory_contract.must_carry_into_next` declared.
  4. For long-horizon variants (ending in PD-long), populates
     `cumulative_gt` so the final wishlist must contain at least one anchor
     per earlier session.
  5. Writes the parent task JSON to `tasks/pool/multisession/`.

The composer DOES NOT regenerate the sub-task JSONs — they must already exist
under `tasks/pool/validated/<SUB>/<task_id>.json`.

Heterogeneous-by-default, with a small homogeneous slice (doc/multisession_design.md §E.4).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Literal, Optional

from ..schema.task_instance import MemoryContract, TaskInstance
from ..schema.multisession import (
    MultiSessionTask,
    SessionRef,
    save_multi_session,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_POOL = REPO_ROOT / "tasks" / "pool" / "validated"
MS_POOL = REPO_ROOT / "tasks" / "pool" / "multisession"


# Variant schema: ordered (sub_task pattern, final_session_kind, count).
HETERO_VARIANTS = {
    "short_3":          (["NC", ["SA", "IC", "VL"], "NC"], False, 30),
    "short_4":          (["NC", "SA", "IC", "VL"], False, 40),
    "short_5":          (["NC", "SA", "IC", "VL", ["NC", "SA"]], False, 30),
    "long_3":           (["NC", "SA", "PD_short"], True, 20),
    "long_4":           (["NC", "SA", "IC", "PD_medium"], True, 20),
    "long_5":           (["NC", "SA", "IC", "VL", "PD_long"], True, 10),
    # Ablation: 4-session heterogeneous mix with NO long horizon. Pulled
    # from §E.4 in the design doc — moved here (not in HOMOGENEOUS_VARIANTS)
    # because the sessions ARE heterogeneous; only the absence of PD makes
    # it an ablation variant.
    "mixed_no_pd_4":    (["NC", "SA", "IC", "VL"], False, 5),
}
HOMOGENEOUS_VARIANTS = {
    "homo_nc_5":   (["NC"] * 5, False, 5),
    "homo_sa_5":   (["SA"] * 5, False, 5),
    "homo_pd":     (["PD_short", "PD_short", "PD_medium"], True, 5),
}


def _load_pool() -> dict[str, list[Path]]:
    """Map sub_task code → sorted list of JSON paths in the validated pool."""
    out: dict[str, list[Path]] = {}
    for sub in ["NC", "SA", "IC", "VL", "PD"]:
        d = VAL_POOL / sub
        out[sub] = sorted(d.glob("*.json")) if d.exists() else []
    # PD-long is stored under PD/ subdir with filename prefix `pd_long_*`.
    # Split by length tier for the composer's variant rules.
    pd_long_short = [p for p in out["PD"] if p.name.startswith("pd_long_short_")]
    pd_long_med = [p for p in out["PD"] if p.name.startswith("pd_long_medium_")]
    pd_long_long = [p for p in out["PD"] if p.name.startswith("pd_long_long_")]
    pd_base = [p for p in out["PD"] if not p.name.startswith("pd_long_")]
    out["PD"] = pd_base
    out["PD_short"] = pd_long_short
    out["PD_medium"] = pd_long_med
    out["PD_long"] = pd_long_long
    return out


def _load_instance(path: Path) -> TaskInstance:
    return TaskInstance.model_validate_json(path.read_text())


def _pick_session_id(pool: list[Path], rng: random.Random, used: set[str]) -> Optional[str]:
    """Pick a sub-task id that hasn't been used in this parent task yet."""
    if not pool:
        return None
    candidates = [p.stem for p in pool if p.stem not in used]
    if not candidates:
        return None
    return rng.choice(candidates)


def _resolve_kind(kind_or_choices: str | list[str], rng: random.Random) -> str:
    if isinstance(kind_or_choices, list):
        return rng.choice(kind_or_choices)
    return kind_or_choices


# Plural → singular noun for the cross-session preamble.
_NOUNS = {
    "rugs": "rug", "chairs": "chair", "lamps": "lamp", "vases": "vase",
    "cushions": "cushion", "tables": "side table",
    "bookshelves": "bookshelf", "plant_pots": "plant pot",
    "sofas": "sofa", "wall_art": "piece of wall art",
}


def _noun_from_category(cat_slug: str) -> str:
    return _NOUNS.get(cat_slug, cat_slug.rstrip("s") if cat_slug.endswith("s") else cat_slug)


def _propagate_retrieves(
    sessions: list[tuple[str, TaskInstance]],
) -> list[TaskInstance]:
    """Walk forward through the composed sessions.

    Decision C (no-accumulation carry): each session N's
    retrieves_from_prior is JUST the anchors of session N-1 — the immediate
    predecessor only. We do NOT OR in older sessions' must_carry. This
    matches the "bridge per session" intent: we test whether the agent can
    recall the most-recent session, not whether the memory pipeline
    accumulates indefinitely.

    Long-horizon "remember session 0 after 4 sessions" is still measured
    via the parent task's `cumulative_gt` (which holds one anchor per
    session), NOT via per-session retrieves_from_prior.
    """
    out: list[TaskInstance] = []
    prev_must_carry: list[str] = []
    for i, (kind, inst) in enumerate(sessions):
        if i == 0:
            updated = inst
        else:
            new_contract = MemoryContract(
                encodes=list(inst.memory_contract.encodes),
                retrieves_from_prior=list(prev_must_carry),  # ← only N-1
                must_carry_into_next=list(inst.memory_contract.must_carry_into_next),
            )
            updated = inst.model_copy(update={"memory_contract": new_contract})
        out.append(updated)
        prev_must_carry = list(updated.memory_contract.must_carry_into_next)
    return out


def compose(
    variant: str,
    seed: int,
    pool: dict[str, list[Path]],
    rng_override: Optional[random.Random] = None,
) -> MultiSessionTask | None:
    rng = rng_override or random.Random(f"ms.{variant}.{seed}")
    spec_table = HETERO_VARIANTS if variant in HETERO_VARIANTS else HOMOGENEOUS_VARIANTS
    if variant not in spec_table:
        raise ValueError(f"unknown variant: {variant!r}")
    pattern, has_long_final, _count = spec_table[variant]

    used_ids: set[str] = set()
    session_kinds: list[str] = [_resolve_kind(p, rng) for p in pattern]
    chosen_paths: list[tuple[str, Path]] = []
    for kind in session_kinds:
        sid = _pick_session_id(pool.get(kind, []), rng, used_ids)
        if sid is None:
            return None
        used_ids.add(sid)
        # Map kind → on-disk sub-task dir
        on_disk_sub = "PD" if kind.startswith("PD") else kind
        chosen_paths.append((kind, VAL_POOL / on_disk_sub / f"{sid}.json"))

    # Load each TaskInstance.
    instances: list[tuple[str, TaskInstance]] = []
    for kind, path in chosen_paths:
        if not path.exists():
            return None
        instances.append((kind, _load_instance(path)))

    # Propagate retrieves_from_prior + must_carry_into_next across sessions.
    propagated = _propagate_retrieves(instances)

    # Cross-session bridging preambles. Session 0 has none. Session N (N>=1)
    # gets a natural-language reference to session N-1's anchor — this is
    # what makes the multi-session task ACTUALLY require memory: without it
    # the agent has nothing in-prompt that demands recall.
    preambles = ["" for _ in propagated]
    for i in range(1, len(propagated)):
        prior_inst = propagated[i - 1]
        prior_noun = _noun_from_category(prior_inst.category_ids[0]
                                          if prior_inst.category_ids else "item")
        current_inst = propagated[i]
        current_noun = _noun_from_category(current_inst.category_ids[0]
                                            if current_inst.category_ids else "item")
        preambles[i] = (
            f"Earlier we looked at a {prior_noun} I liked — keep its style/tone in mind. "
            f"Now for the next item, I want a {current_noun} that fits the same tone as that {prior_noun}. "
        )

    # Build SessionRefs from propagated instances. The contract changes
    # produced by _propagate_retrieves are stored on the SessionRef
    # overrides — NOT persisted back to the sub-task JSON.
    refs = [
        SessionRef(
            task_id=propagated[i].task_id,
            sub_task=propagated[i].sub_task,
            order_index=i,
            retrieves_from_prior_override=list(propagated[i].memory_contract.retrieves_from_prior),
            must_carry_override=list(propagated[i].memory_contract.must_carry_into_next),
            cross_session_user_preamble=preambles[i],
        )
        for i in range(len(propagated))
    ]

    cumulative_turn_budget = sum(len(p.turns) for p in propagated)

    cumulative_gt: list[str] = []
    if has_long_final:
        # Long-horizon: cumulative wishlist GT = one anchor per non-final
        # session (their must_carry_into_next' head). The final PD session's
        # own GT is added as a "must include" too — this is the "cumulative
        # state" decision from doc/multisession_design.md §F.2.
        for p in propagated[:-1]:
            if p.memory_contract.must_carry_into_next:
                cumulative_gt.append(p.memory_contract.must_carry_into_next[0])
        final = propagated[-1]
        if final.ground_truth.target_variant_id:
            cumulative_gt.append(final.ground_truth.target_variant_id)

    homogeneous = variant.startswith("homo_")
    parent_id = f"ms_{variant}_s{seed:04d}"
    return MultiSessionTask(
        task_id=parent_id,
        variant=variant,
        sessions=refs,
        cumulative_turn_budget=cumulative_turn_budget,
        cumulative_gt=cumulative_gt,
        homogeneous=homogeneous,
        metadata={
            "session_kinds": session_kinds,
            "session_task_ids": [r.task_id for r in refs],
        },
    )


def bulk_compose(
    out_dir: Path = MS_POOL,
    seed_base: int = 0,
) -> list[Path]:
    """Compose the full 150-task grid (heterogeneous + homogeneous)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = _load_pool()
    written: list[Path] = []
    seed = seed_base
    for variant, (_, _, count) in {**HETERO_VARIANTS, **HOMOGENEOUS_VARIANTS}.items():
        for _ in range(count):
            ms = compose(variant, seed, pool)
            seed += 1
            if ms is None:
                continue
            written.append(save_multi_session(ms, pool=out_dir))
    return written


if __name__ == "__main__":
    import sys
    paths = bulk_compose()
    print(f"composed {len(paths)} multi-session tasks", file=sys.stderr)
    for p in paths[:5]:
        print(p)
