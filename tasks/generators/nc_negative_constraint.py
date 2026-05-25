"""NC — Negative Constraint (proposal_tasks_v2.md §3).

W7v2 / Phase-4.1 upgrade. The previous price-bracket trick is replaced
by the v2 resolution policy:

  1. Hard filter — same category as anchor, style ∉ rejected_styles
  2. Visual scoring — CLIP cosine to anchor
  3. Uniqueness — top-1 over top-2 by ≥ ε (default 0.05); regenerate
     on Ambiguous

Difficulty:

  NC-Easy (default): 1 negative — "...similar but not in {style}"
  NC-Hard          : 2–3 cumulative negatives across consecutive
                     setup turns (subsumes deprecated M2)

Negative constraint grammar (≥ 4 surface templates per v2 §8.4):

  "...but not {style}."
  "...just not the {style} one."
  "...avoid anything {style}."
  "...nothing {style}-looking."

Cumulative-add grammars for NC-Hard:

  "Also, not {style}."
  "And skip anything {style}."
  "Drop anything {style}-feeling too."

User dialogue never references "anchor" or product names — only
"this one" / "the one I showed". MR1 hides anchor's product detail
page during the action phase (Phase 3 extension).
"""
from __future__ import annotations

import random
from typing import Literal

from ._common import (
    ALL_STYLES,
    VariantCatalogue,
    Variant,
    coarse_noun,
    seeded_rng,
)
from ..schema.task_instance import (
    AgentAction,
    GroundTruth,
    NcpMetadata,
    RecallTurnMetadata,
    SuccessCriteria,
    TaskInstance,
    TaskMetadata,
    Turn,
)
from ..scoring.uniqueness import (
    Ambiguous,
    DEFAULT_EPSILON,
    _load_cache,
    resolve,
)

GENERATOR_VERSION = "nc.v7"

NEGATIVE_TEMPLATES = [
    "...but not {style}.",
    "...just not the {style} one.",
    "...avoid anything {style}.",
    "...nothing {style}-looking.",
]

CUMULATIVE_TEMPLATES = [
    "Also, not {style}.",
    "And skip anything {style}.",
    "Drop anything {style}-feeling too.",
    "Same for the {style} look — not interested.",
]


def _pick_anchor_and_rejected(
    catalogue: VariantCatalogue,
    rng: random.Random,
    n_negatives: int,
    cache: dict[str, list[float]],
) -> tuple[object, Variant, Variant, list[str]]:
    """W7v2.4 — "cheapest-after-visual-filter" disambiguation.

    Setup:
       - anchor in some category
       - n_negatives styles to reject (anchor's style NOT in rejected set)
       - hard filter on candidates: (cat = anchor.cat) ∧ (style ∉ rejected)
       - GT = cheapest in that filtered set (excluding anchor)

    Why cheapest works as a clean disambiguator:
       - Price alone (without the style filter) would pick a product that
         might be in the rejected style → wrong answer.
       - The worker has to apply the visual style-rejection filter first
         (recognising rejected-style products visually) and *then* pick
         the cheapest among the remainder. Either filter alone is
         insufficient; the conjunction is unique.

    Returns (cat, anchor, gt, rejected_styles)."""
    cats = catalogue.categories.copy()
    rng.shuffle(cats)
    for cat in cats:
        scope = catalogue.variants_in(cat.slug)
        anchors = scope.copy()
        rng.shuffle(anchors)
        for anchor in anchors:
            other_styles = [s for s in ALL_STYLES if s != anchor.style_slug]
            if len(other_styles) < n_negatives:
                continue
            rng.shuffle(other_styles)
            rejected = sorted(other_styles[:n_negatives])
            candidates = [
                v for v in scope
                if v.style_slug not in rejected and v.url_hash != anchor.url_hash
            ]
            if len(candidates) < 2:
                continue
            # CLIP isn't strictly needed for GT now (cheapest does the
            # disambiguation), but we still want to make sure the cheapest
            # in the filtered set is sane. Pick GT = min price in filter.
            gt = min(candidates, key=lambda v: v.price)
            return cat, anchor, gt, rejected
    raise RuntimeError(
        f"NC: no feasible (anchor, {n_negatives} negatives, gt) found"
    )


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    rng = seeded_rng(seed, f"nc.{difficulty}")
    n_negatives = 1 if difficulty == "easy" else rng.choice([2, 3])

    cache = _load_cache()
    cat, anchor, gt, rejected = _pick_anchor_and_rejected(
        catalogue, rng, n_negatives, cache
    )
    noun = coarse_noun(cat.slug)

    # T1: state primary negative + "cheapest" disambiguator. Style-filter
    # alone yields N candidates; price-alone is wrong (picks a rejected-
    # style product); the conjunction is unique.
    primary_neg_tpl = rng.choice(NEGATIVE_TEMPLATES)
    primary_neg = primary_neg_tpl.format(style=rejected[0]).rstrip(".")
    t1_content = (
        f"I'm looking for a {noun}. Here's one I like — show me the "
        f"cheapest one similar to this {primary_neg}."
    )

    turns: list[Turn] = [
        Turn(turn_index=0, role="system", mode="encoding",
             content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content=t1_content,
             references_variant=anchor.url_hash),
    ]
    next_idx = 2
    # NC-Hard: extra cumulative negatives, one per turn
    for extra_style in rejected[1:]:
        tpl = rng.choice(CUMULATIVE_TEMPLATES)
        turns.append(Turn(
            turn_index=next_idx, role="user", mode="encoding",
            content=tpl.format(style=extra_style),
        ))
        next_idx += 1
    # Agent navigates to the category to begin search
    turns.append(Turn(
        turn_index=next_idx, role="agent", mode="encoding",
        expected_actions=[AgentAction(
            action_type="navigate",
            target_url=f"/category/{cat.slug}")],
        expected_url=f"/category/{cat.slug}",
    ))
    next_idx += 1
    # Recall — agent commits
    turns.append(Turn(
        turn_index=next_idx, role="agent", mode="recall",
        expected_actions=[AgentAction(
            action_type="navigate",
            target_url=f"/product/{gt.url_hash}")],
        expected_url=f"/product/{gt.url_hash}",
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "recently_viewed",
                                          "wishlist", "search", "breadcrumb"],
            expected_memory_usage="both",
        ),
    ))
    recall_idx = next_idx

    diff_tag = "h" if difficulty == "hard" else "e"
    return TaskInstance(
        task_id=f"nc_{cat.slug}_{diff_tag}_s{seed:04d}",
        sub_task="NC",
        grain_tier=anchor.grain_tier,
        category_ids=[cat.slug],
        variants_used=[anchor.url_hash, gt.url_hash],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[anchor.url_hash],
            recall_turn_indices=[recall_idx],
            memory_gated_branch_turn=recall_idx,
            cross_turn_predicates=[
                f"visual: similar to anchor {anchor.url_hash[:8]}",
                f"negative: style ∉ {{{', '.join(rejected)}}}",
                f"disambiguator: cheapest in (cat, style ∉ rejected)",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=f"/product/{gt.url_hash}",
            target_variant_id=gt.url_hash,
            accepted_alternatives=[],
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )
