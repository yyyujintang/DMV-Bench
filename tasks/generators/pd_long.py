"""PD-long — Preference Drift with extended filler chain.

Wraps the standard PD generator and injects N filler turns mid-task to
hit one of three length tiers (30 / 80 / 150 turns). The original PD's
preference state, drift patterns, and GT-resolver logic are unchanged;
we only stretch the dialog by interleaving more distractor / aborted-
exploration / revisit cycles between the original Phase-2 and Phase-3.

The cumulative-state GT (multisession_design.md §F.2) is layered on top
by the multi-session composer, NOT here — this generator still produces
a single-target url_match GT identical to standard PD.

Length tiers:
  - tier 30: 15 → 30 turns (~15 extra)
  - tier 80: 15 → 80 turns (~65 extra)
  - tier 150: 15 → 150 turns (~135 extra)
"""
from __future__ import annotations

import random
from typing import Literal

from ._common import ALL_STYLES, VariantCatalogue, Variant, coarse_noun, seeded_rng
from ..schema.task_instance import (
    AgentAction,
    TaskInstance,
    TaskMetadata,
    Turn,
)
from .pd_preference_drift import (
    DISTRACTOR_POOL,
    STYLE_DESCRIPTOR,
    generate as generate_pd_base,
)

GENERATOR_VERSION = "pd_long.v1"

LENGTH_TIERS = {
    "short": 30,
    "medium": 80,
    "long": 150,
}


REVISIT_TEMPLATES = [
    "Can you show me that {style} one again?",
    "Let me see the {style} options once more.",
    "What about going back to the {style} look?",
    "Could you pull up that {style} piece?",
]
EXPLORE_TEMPLATES = [
    "Wait, briefly — what would {style} look like?",
    "Out of curiosity, do you have anything {style}?",
    "Quick aside — any {style} options?",
    "What if we went {style} instead?",
]
ABORT_TEMPLATES = [
    "Actually, never mind, scrap the {style} idea.",
    "Forget I asked about {style}.",
    "Drop the {style} thread — let's keep going.",
    "Ignore the {style} detour.",
]


def _build_filler_block(
    cat_slug: str,
    other_styles: list[str],
    rng: random.Random,
    n_pairs: int,
    start_turn_index: int,
) -> list[Turn]:
    """Generate `n_pairs` (user, agent) filler turn pairs that don't touch
    the active preference state but stretch the dialog horizon.

    Each pair is one of:
      A) distractor question + agent answer
      B) explore-and-abort style branch (user proposes style → agent navigates
         to that style's collection → user retracts)
    """
    turns: list[Turn] = []
    i = start_turn_index
    for k in range(n_pairs):
        kind = rng.choice(["distractor", "explore_abort", "distractor"])  # 2/3 distractor
        if kind == "distractor":
            user_msg = rng.choice(DISTRACTOR_POOL)
            turns.append(Turn(turn_index=i, role="user", mode="encoding", content=user_msg))
            i += 1
            turns.append(Turn(
                turn_index=i, role="agent", mode="encoding",
                content="Let me share the details on that.",
            ))
            i += 1
        else:
            style = rng.choice(other_styles)
            other_bare = STYLE_DESCRIPTOR[style].lstrip("a ").lstrip("an ")
            tpl_explore = rng.choice(EXPLORE_TEMPLATES)
            turns.append(Turn(
                turn_index=i, role="user", mode="encoding",
                content=tpl_explore.format(style=other_bare),
            ))
            i += 1
            turns.append(Turn(
                turn_index=i, role="agent", mode="encoding",
                expected_actions=[AgentAction(
                    action_type="navigate",
                    target_url=f"/collection/{cat_slug}-{style}",
                )],
                expected_url=f"/collection/{cat_slug}-{style}",
            ))
            i += 1
            turns.append(Turn(
                turn_index=i, role="user", mode="encoding",
                content=rng.choice(ABORT_TEMPLATES).format(style=other_bare),
            ))
            i += 1
            # Pair count: we treat this as one "pair" of weight 1.5 but
            # we still advance n_pairs by 1 — slight overshoot is fine
            # because the final length is approximate.
    return turns


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    length: Literal["short", "medium", "long"] = "short",
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    """Build a long PD instance by wrapping standard PD with filler turns.

    The injection point is just before the standard PD's Phase-3
    distractor section (turn 8 in the base 15-turn task). All later turn
    indices are shifted up by the number of inserted turns.
    """
    rng = seeded_rng(seed, f"pd_long.{length}.{difficulty}")
    base = generate_pd_base(catalogue, seed, difficulty=difficulty)
    target_total = LENGTH_TIERS[length]
    n_extra = max(0, target_total - len(base.turns))
    # n_pairs ≈ n_extra / 2 (most blocks are 2-turn distractor pairs;
    # explore_abort is 3 turns but we keep the estimate close).
    n_pairs = max(0, n_extra // 2)

    INJECT_AT = 8  # before the original Phase-3 distractor
    cat_slug = base.category_ids[0]
    # Pick styles other than the one currently active.
    other_styles = [s for s in ALL_STYLES]
    rng.shuffle(other_styles)

    filler = _build_filler_block(
        cat_slug=cat_slug,
        other_styles=other_styles,
        rng=rng,
        n_pairs=n_pairs,
        start_turn_index=INJECT_AT,
    )
    shift = len(filler)

    new_turns: list[Turn] = []
    for t in base.turns:
        if t.turn_index < INJECT_AT:
            new_turns.append(t)
        else:
            new_turns.append(t.model_copy(update={"turn_index": t.turn_index + shift}))
    # Inject filler at INJECT_AT (already indexed correctly).
    new_turns = (
        [t for t in new_turns if t.turn_index < INJECT_AT]
        + filler
        + [t for t in new_turns if t.turn_index >= INJECT_AT + shift]
    )
    new_turns.sort(key=lambda x: x.turn_index)

    # Update recall_turn_indices + memory_gated_branch_turn by the shift.
    shifted_recall = [
        idx + shift if idx >= INJECT_AT else idx
        for idx in base.ncp_metadata.recall_turn_indices
    ]
    shifted_branch = (
        base.ncp_metadata.memory_gated_branch_turn + shift
        if base.ncp_metadata.memory_gated_branch_turn >= INJECT_AT
        else base.ncp_metadata.memory_gated_branch_turn
    )

    diff_tag = "h" if difficulty == "hard" else "e"
    new_id = f"pd_long_{length}_{cat_slug}_{diff_tag}_s{seed:04d}"
    return base.model_copy(update={
        "task_id": new_id,
        "turns": new_turns,
        "ncp_metadata": base.ncp_metadata.model_copy(update={
            "recall_turn_indices": shifted_recall,
            "memory_gated_branch_turn": shifted_branch,
            "cross_turn_predicates": list(base.ncp_metadata.cross_turn_predicates) + [
                f"length_tier: {length} (target {target_total} turns)"
            ],
        }),
        "metadata": TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    })
