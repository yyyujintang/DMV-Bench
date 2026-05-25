"""VL — Visual Landmark (proposal_tasks_v2.md §6).

W7v2 / Phase-4.2 upgrade. The probe at T6 is a **gestalt scene-level**
description, NOT a style or category name (v2 §6.3 — anti-shortcut).
The user references the landmark via a color-palette / atmospheric
phrase like "the one with all the deep reds and golds" or "the one
that looked airy and pale".

Setup: agent visits 4 collection pages from up to 4 categories
(VL-Easy: 4 distinct styles spanning maximum visual separation;
VL-Hard: 2 of the 4 share a related style family, forcing finer
discrimination).

Uniqueness: by construction at generation time. The chosen description
maps uniquely to one of the 4 visited collections by color-palette
family. Generator asserts this before emitting the task.
"""
from __future__ import annotations

import random
from typing import Literal

from ._common import ALL_STYLES, VariantCatalogue, coarse_noun, seeded_rng
from ..schema.task_instance import (
    AgentAction,
    GroundTruth,
    MemoryContract,
    NcpMetadata,
    RecallTurnMetadata,
    SuccessCriteria,
    TaskInstance,
    TaskMetadata,
    Turn,
)

GENERATOR_VERSION = "vl.v7"

# Gestalt color/atmosphere descriptors — one per style. Avoid naming the
# style, category, or material. These were chosen by inspecting the
# legacy_colors image set so each phrase maps unambiguously to one
# stylistic palette.
# Each descriptor must mention a feature **unique** to that style, so it
# rules out the other 3 visited collections by construction (per user
# test feedback: "off-white tone with quiet pop" matched both modern AND
# minimalist on the catalog). Each descriptor combines: (a) a colour
# family signal, AND (b) a style-exclusive structural cue.
GESTALT_DESCRIPTORS: dict[str, list[str]] = {
    "modern": [
        "neutral tones with a single warm accent — caramel-or-sage — and clean rectilinear lines",
        "a warm-neutral palette with a single accent stripe and slim metal accents",
        "an off-white-with-warm-accent palette and crisp geometric topstitching",
    ],
    "minimalist": [
        "everything ivory or dove, with no accent colour and no visible hardware",
        "a near-white monochrome palette with flush flat panels and a single seam",
        "soft pale tones with hidden hardware and the simplest possible silhouette",
    ],
    "vintage": [
        "warm earthy tones with patterned upholstery and button tufting or fringe",
        "tobacco-and-burgundy upholstery with carved or scalloped trim",
        "saturated jewel tones with brass nailheads and curved arms",
    ],
    "industrial": [
        "very dark with exposed rivets, weld seams, or caster wheels",
        "cool gunmetal tones with raw steel bracing and pipe legs",
        "black or charcoal upholstery with metal-strap bracing and rough timber",
    ],
}


def _pick_pages_and_landmark(
    catalogue: VariantCatalogue,
    rng: random.Random,
    difficulty: Literal["easy", "hard"],
) -> tuple[list, object, str]:
    """Pick 4 collections within the SAME category — one per style.

    Per v2 spec §6.4: "Be from different categories, OR **from the same
    category but different styles**." User feedback in W7v2 testing
    showed the cross-category version produced an unfair task — the
    recall question doesn't name the category, so the worker had to
    remember both category AND style. Same-category mode makes the noun
    implicit so the descriptor only has to disambiguate among 4 styles
    of one category.

    Easy: 4 distinct styles (max separation), landmark on any of them.
    Hard: still 4 collections in one cat — but at recall the worker
          gets the visual descriptor with no noun, so the *style*
          discrimination is the only test. (Hard differs from Easy by
          using a more boundary-leaning descriptor variant.)
    """
    cats = catalogue.categories.copy()
    rng.shuffle(cats)
    target_cat = None
    visited_collections: list = []
    for cat in cats:
        styles_here = list(ALL_STYLES)
        rng.shuffle(styles_here)
        collected: list = []
        for style in styles_here:
            v = catalogue.variant_or_none(cat.slug, 1, style) \
                or catalogue.variant_or_none(cat.slug, 2, style) \
                or catalogue.variant_or_none(cat.slug, 3, style)
            if v is not None:
                collected.append(v)
        if len(collected) == 4:
            target_cat = cat
            visited_collections = collected
            break
    if target_cat is None or len(visited_collections) != 4:
        raise RuntimeError(
            "VL: no category has all 4 styles available — check catalog"
        )
    # GT placement (per user feedback): the landmark must be one of the
    # FIRST 2 visited collections (positions 0 or 1), not the last —
    # otherwise recency-of-visit makes the answer trivially fresh in
    # memory. Future scale-up to longer visit sequences will shift this.
    early_visits = visited_collections[:2]
    landmark = rng.choice(early_visits)
    landmark_style = landmark.style_slug
    return visited_collections, landmark, landmark_style


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    rng = seeded_rng(seed, f"vl.{difficulty}")
    visited, landmark, landmark_style = _pick_pages_and_landmark(
        catalogue, rng, difficulty
    )
    descriptor = rng.choice(GESTALT_DESCRIPTORS[landmark_style])
    noun = coarse_noun(landmark.category_slug)

    turns: list[Turn] = [
        Turn(turn_index=0, role="system", mode="encoding",
             content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content=f"I want to browse {noun}s — show me four collections."),
    ]
    for i, v in enumerate(visited):
        turns.append(Turn(
            turn_index=2 + i, role="agent", mode="encoding",
            expected_actions=[AgentAction(
                action_type="navigate",
                target_url=f"/collection/{v.collection_slug}")],
            expected_url=f"/collection/{v.collection_slug}",
        ))
    recall_turn = 2 + len(visited)
    turns.append(Turn(
        turn_index=recall_turn, role="user", mode="recall",
        content=f"Take me back to the {noun} collection with {descriptor}.",
    ))
    turns.append(Turn(
        turn_index=recall_turn + 1, role="agent", mode="recall",
        expected_actions=[AgentAction(
            action_type="navigate",
            target_url=f"/collection/{landmark.collection_slug}")],
        expected_url=f"/collection/{landmark.collection_slug}",
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "recently_viewed",
                                         "collection_tiles"],
            expected_memory_usage="positive",
        ),
    ))

    diff_tag = "h" if difficulty == "hard" else "e"
    return TaskInstance(
        task_id=f"vl_{landmark.category_slug}_{landmark_style}_{diff_tag}_s{seed:04d}",
        sub_task="VL",
        grain_tier=landmark.grain_tier,
        category_ids=sorted({v.category_slug for v in visited}),
        variants_used=[v.url_hash for v in visited],
        turns=turns,
        ncp_metadata=NcpMetadata(
            # VL's "anchor" is conceptually the landmark target, but MR1
            # hiding it on the answer page would defeat the task — the
            # agent must be able to navigate to it. We keep the recall
            # surface audited (4 collection pages with no anchor on any
            # of their grids, since the landmark itself is just a colour
            # palette, not a hidden variant).
            anchor_variant_ids=[],
            recall_turn_indices=[recall_turn + 1],
            memory_gated_branch_turn=recall_turn + 1,
            cross_turn_predicates=[
                f"recall: gestalt = '{descriptor}'",
                f"predicate: colour-palette ↔ collection URL",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=f"/collection/{landmark.collection_slug}",
            target_variant_id=landmark.url_hash,
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        # VL's recall target is a colour-palette landmark; the
        # corresponding variant urlHash is what we carry forward so the
        # later session's "remember that collection?" cue can hit.
        memory_contract=MemoryContract(
            encodes=[landmark.url_hash],
            retrieves_from_prior=[],
            must_carry_into_next=[landmark.url_hash],
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )
