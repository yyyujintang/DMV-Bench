"""SA — Style Abstraction (proposal_tasks_v2.md §4).

W7v2 / Phase-4.4 upgrade. Replaces price-bracket isolation with the v2
resolution policy:

  1. Hard filter — products in target_category, in the shared style
  2. Visual scoring — CLIP cosine to the centroid of 3 exemplar
     embeddings
  3. Uniqueness ε margin check; regen on Ambiguous

Setup: user shows 3 exemplars from 3 distinct categories, all sharing
one style (never named in chat). Agent must extract the shared
signature and find a 4th product in a new category.

SA-Easy: 3 prototypical exemplars (close to style centroid by CLIP);
          target category has 1 strong match.
SA-Hard: 1 of 3 exemplars sits on the style boundary (low CLIP cosine
          to the style centroid); target category has 2 candidate
          collections — only one passes ε margin.
"""
from __future__ import annotations

import random
from typing import Literal

from ._common import ALL_STYLES, VariantCatalogue, Variant, coarse_noun, seeded_rng
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
    cosine,
    resolve,
)

GENERATOR_VERSION = "sa.v8"


def _pick_exemplars_target(
    catalogue: VariantCatalogue,
    rng: random.Random,
    cache: dict[str, list[float]],
    difficulty: Literal["easy", "hard"],
) -> tuple[list[Variant], object, Variant]:
    """Pick (3 exemplars across 3 cats, target_cat, GT) such that:
       - all 3 exemplars share a style
       - GT is in a 4th category, same style, **cheapest of its style**
       - CLIP resolve with centroid query passes ε margin (sanity check)"""
    # W7v2.2: flatten and shuffle (cat, style) pairs as a single list to
    # avoid the residual style-first convergence (4/4 → plant_pots/
    # industrial under cat-first). Each seed now explores a different
    # random walk over all 40 (cat, style) pairs.
    pairs = [(c, s) for c in catalogue.categories for s in ALL_STYLES]
    rng.shuffle(pairs)
    cats = catalogue.categories
    for tcat, style in pairs:
            target_pool = [
                v for v in catalogue.variants_in(tcat.slug)
                if v.style_slug == style
            ]
            if not target_pool:
                continue
            anchor_cats = [
                c for c in cats
                if c.slug != tcat.slug
                and any(v.style_slug == style for v in catalogue.variants_in(c.slug))
            ]
            if len(anchor_cats) < 3:
                continue
            rng.shuffle(anchor_cats)
            ac = anchor_cats[:3]
            anchors: list[Variant] = []
            for c in ac:
                pool = [v for v in catalogue.variants_in(c.slug)
                        if v.style_slug == style]
                if difficulty == "hard" and len(anchors) == 0 and len(pool) > 1:
                    # Boundary exemplar: pick the one furthest from
                    # the others (max intra-pool variance proxy)
                    style_centroid = _style_centroid(catalogue, style, cache)
                    pool_sorted = sorted(
                        pool, key=lambda v: cosine(cache[v.url_hash], style_centroid)
                    )
                    anchors.append(pool_sorted[0])  # least prototypical
                else:
                    anchors.append(rng.choice(pool))
            # Now scoring against centroid of these 3 exemplars
            centroid_query = [a.url_hash for a in anchors]
            candidates = [v.url_hash for v in target_pool]
            if difficulty == "hard":
                # Include distractor candidates from a *different* style
                # in the target category (forces visual discrimination)
                distractor_style_pool = [
                    v.url_hash for v in catalogue.variants_in(tcat.slug)
                    if v.style_slug != style
                ]
                rng.shuffle(distractor_style_pool)
                candidates = candidates + distractor_style_pool[:3]
            try:
                result = resolve(
                    candidates, centroid_query,
                    cache=cache, epsilon=DEFAULT_EPSILON,
                )
            except Ambiguous:
                continue
            # W7v2.4 — drop CLIP for GT pick. GT = cheapest in (target_cat,
            # matching_style). The shared style is the visual constraint
            # (inferred from 3 anchors); price-alone in target cat picks
            # a non-matching-style product → wrong. Conjunction is unique.
            target_style_pool = [
                v for v in catalogue.variants_in(tcat.slug)
                if v.style_slug == style
            ]
            if not target_style_pool:
                continue
            gt = min(target_style_pool, key=lambda v: v.price)
            return anchors, tcat, gt
    raise RuntimeError(f"SA-{difficulty}: no feasible (exemplars, target, gt) found")


def _style_centroid(
    catalogue: VariantCatalogue,
    style: str,
    cache: dict[str, list[float]],
) -> list[float]:
    members = [
        cache[v.url_hash] for v in catalogue.variants
        if v.style_slug == style and v.url_hash in cache
    ]
    if not members:
        raise RuntimeError(f"no embeddings for style {style}")
    dim = len(members[0])
    acc = [0.0] * dim
    for v in members:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(members) for x in acc]


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    rng = seeded_rng(seed, f"sa.{difficulty}")
    cache = _load_cache()
    anchors, tcat, gt = _pick_exemplars_target(catalogue, rng, cache, difficulty)
    target_noun = coarse_noun(tcat.slug)

    turns: list[Turn] = [
        Turn(turn_index=0, role="system", mode="encoding",
             content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content="Here are three things I like. Notice anything about my taste?",
             references_variant=anchors[0].url_hash),
    ]
    for i, a in enumerate(anchors):
        turns.append(Turn(
            turn_index=2 + i, role="agent", mode="encoding",
            expected_actions=[AgentAction(
                action_type="navigate",
                target_url=f"/product/{a.url_hash}")],
            expected_url=f"/product/{a.url_hash}",
        ))
    next_idx = 2 + len(anchors)
    turns.append(Turn(
        turn_index=next_idx, role="user", mode="encoding",
        content=f"Find me the cheapest {target_noun} that fits the same vibe.",
    ))
    next_idx += 1
    # Recall — navigate to target collection then commit to product
    turns.append(Turn(
        turn_index=next_idx, role="agent", mode="recall",
        expected_actions=[AgentAction(
            action_type="navigate",
            target_url=f"/collection/{gt.collection_slug}")],
        expected_url=f"/collection/{gt.collection_slug}",
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "wishlist",
                                          "recently_viewed"],
            expected_memory_usage="positive",
        ),
    ))
    next_idx += 1
    turns.append(Turn(
        turn_index=next_idx, role="agent", mode="recall",
        expected_actions=[AgentAction(
            action_type="navigate",
            target_url=f"/product/{gt.url_hash}")],
        expected_url=f"/product/{gt.url_hash}",
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "wishlist"],
            expected_memory_usage="positive",
        ),
    ))

    diff_tag = "h" if difficulty == "hard" else "e"
    return TaskInstance(
        task_id=f"sa_{tcat.slug}_{gt.style_slug}_{diff_tag}_s{seed:04d}",
        sub_task="SA",
        grain_tier=gt.grain_tier,
        category_ids=sorted({a.category_slug for a in anchors} | {tcat.slug}),
        variants_used=[a.url_hash for a in anchors] + [gt.url_hash],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[a.url_hash for a in anchors],
            recall_turn_indices=[next_idx - 1, next_idx],
            memory_gated_branch_turn=next_idx,
            cross_turn_predicates=[
                f"visual: shared style centroid of 3 exemplars",
                f"category: {tcat.slug} (disjoint from exemplar cats)",
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
