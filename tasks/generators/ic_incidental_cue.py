"""IC — Incidental Cue (proposal_tasks_v2.md §5).

W7v2 / Phase-4.3 upgrade. Replaces the legacy page-overlay W3 cue
system with **product-baked** incidental details
(ProductVariant.incidentalDetails, tagged at seed time from a
style-conditioned vocabulary in tools/tag_incidental_details.py).

Setup: agent visits 4 product pages. **Exactly one** carries the cue
referenced at recall. The other 3 carry different details, so the
agent can't simply look for "the page that had any notable detail".

IC-Easy: 4 product detail pages, recall references a specific detail.
IC-Hard: 4 collection pages, where the cue belongs to one specific
variant card on one of the four collection pages.

Generator invariant (per user-locked override 4): after sampling the
4-page sequence, programmatically assert the target detail appears in
exactly 1 variant across all visited pages. Regenerate on failure.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
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
    MemoryContract,
    NcpMetadata,
    RecallTurnMetadata,
    SuccessCriteria,
    TaskInstance,
    TaskMetadata,
    Turn,
)

GENERATOR_VERSION = "ic.v2"

DETAIL_PHRASE: dict[str, str] = {
    "slim_metal_legs":       "slim metal legs",
    "geometric_topstitching": "geometric topstitching",
    "splayed_legs":          "splayed wooden legs",
    "brass_nailheads":       "brass nailheads",
    "fringe":                "a fringe trim",
    "button_tufting":        "button tufting",
    "ornate_carving":        "ornate carving",
    "exposed_rivets":        "exposed rivets",
    "raw_steel_bracing":     "raw steel bracing",
    "pipe_legs":             "pipe legs",
    "metal_caster_wheels":   "metal caster wheels",
    "hidden_hardware":       "no visible hardware",
    "single_seam":           "a single continuous seam",
    "flush_panels":          "flush flat panels",
}

REPO = Path(__file__).resolve().parents[2]
PRICING_PATH = REPO / "env" / "scripts" / "pricing_naming.json"


def _load_variant_details() -> dict[str, list[str]]:
    data = json.loads(PRICING_PATH.read_text())
    return {h: v.get("incidentalDetails", []) for h, v in data["variants"].items()}


def _pick_easy(
    catalogue: VariantCatalogue,
    rng: random.Random,
    details: dict[str, list[str]],
) -> tuple[list[Variant], Variant, str]:
    cats = catalogue.categories.copy()
    rng.shuffle(cats)
    for cat in cats:
        scope = catalogue.variants_in(cat.slug)
        if len(scope) < 4:
            continue
        rng.shuffle(scope)
        styles_seen: set[str] = set()
        picked: list[Variant] = []
        for v in scope:
            if v.style_slug in styles_seen:
                continue
            picked.append(v)
            styles_seen.add(v.style_slug)
            if len(picked) == 4:
                break
        if len(picked) < 4:
            continue
        unique_pairs: list[tuple[Variant, str]] = []
        for v in picked:
            for d in details.get(v.url_hash, []):
                if sum(1 for w in picked if d in details.get(w.url_hash, [])) == 1:
                    unique_pairs.append((v, d))
        if not unique_pairs:
            continue
        target, target_detail = rng.choice(unique_pairs)
        return picked, target, target_detail
    raise RuntimeError("IC-Easy: no feasible (4 pages, unique detail) found")


def _pick_hard(
    catalogue: VariantCatalogue,
    rng: random.Random,
    details: dict[str, list[str]],
) -> tuple[list[Variant], Variant, str]:
    cats = catalogue.categories.copy()
    rng.shuffle(cats)
    styles = list(ALL_STYLES)
    rng.shuffle(styles)
    visited_collections: list[Variant] = []
    used_cats: set[str] = set()
    for cat in cats:
        if cat.slug in used_cats:
            continue
        style = styles[len(visited_collections) % len(styles)]
        v = catalogue.variant_or_none(cat.slug, 1, style) \
            or catalogue.variant_or_none(cat.slug, 2, style) \
            or catalogue.variant_or_none(cat.slug, 3, style)
        if v is None:
            continue
        used_cats.add(cat.slug)
        visited_collections.append(v)
        if len(visited_collections) == 4:
            break
    if len(visited_collections) < 4:
        raise RuntimeError("IC-Hard: not enough categories")
    all_visited_variants: list[Variant] = []
    for col_rep in visited_collections:
        all_visited_variants.extend(catalogue.variants_in_collection(col_rep.collection_slug))
    unique_pairs: list[tuple[Variant, str]] = []
    for v in all_visited_variants:
        for d in details.get(v.url_hash, []):
            if sum(1 for w in all_visited_variants if d in details.get(w.url_hash, [])) == 1:
                unique_pairs.append((v, d))
    if not unique_pairs:
        raise RuntimeError("IC-Hard: no unique-across-4-collections detail")
    target_variant, target_detail = rng.choice(unique_pairs)
    return visited_collections, target_variant, target_detail


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    rng = seeded_rng(seed, f"ic.{difficulty}")
    details = _load_variant_details()
    if difficulty == "easy":
        visited, target, target_detail = _pick_easy(catalogue, rng, details)
        target_url = f"/product/{target.url_hash}"
        navigate_urls = [f"/product/{v.url_hash}" for v in visited]
    else:
        visited, target, target_detail = _pick_hard(catalogue, rng, details)
        navigate_urls = [f"/collection/{v.collection_slug}" for v in visited]
        target_url = f"/collection/{target.collection_slug}"

    if difficulty == "easy":
        scope = visited
    else:
        scope = []
        for col_rep in visited:
            scope.extend(catalogue.variants_in_collection(col_rep.collection_slug))
    n_with_detail = sum(1 for v in scope if target_detail in details.get(v.url_hash, []))
    if n_with_detail != 1:
        raise AssertionError(
            f"IC invariant violated: detail {target_detail!r} appears "
            f"{n_with_detail}x in scope (expected 1)"
        )

    phrase = DETAIL_PHRASE[target_detail]
    noun = coarse_noun(visited[0].category_slug)
    surface_label = "page" if difficulty == "easy" else "collection"

    turns: list[Turn] = [
        Turn(turn_index=0, role="system", mode="encoding",
             content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content=f"Just browsing — show me around. Take me through a few {noun}s."),
    ]
    for i, url in enumerate(navigate_urls):
        turns.append(Turn(
            turn_index=2 + i, role="agent", mode="encoding",
            expected_actions=[AgentAction(action_type="navigate", target_url=url)],
            expected_url=url,
        ))
    recall_turn = 2 + len(navigate_urls)
    turns.append(Turn(
        turn_index=recall_turn, role="user", mode="recall",
        content=f"Earlier I saw the one with {phrase}. Can you take me back to that {surface_label}?",
    ))
    turns.append(Turn(
        turn_index=recall_turn + 1, role="agent", mode="recall",
        expected_actions=[AgentAction(action_type="navigate", target_url=target_url)],
        expected_url=target_url,
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "recently_viewed",
                                          "wishlist", "history_panel_thumbnails"],
            expected_memory_usage="positive",
        ),
    ))

    diff_tag = "h" if difficulty == "hard" else "e"
    return TaskInstance(
        task_id=f"ic_{visited[0].category_slug}_{target_detail}_{diff_tag}_s{seed:04d}",
        sub_task="IC",
        grain_tier=target.grain_tier,
        category_ids=sorted({v.category_slug for v in visited}),
        variants_used=[v.url_hash for v in visited],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[],
            recall_turn_indices=[recall_turn + 1],
            memory_gated_branch_turn=recall_turn + 1,
            cross_turn_predicates=[
                f"recall: the page with {phrase}",
                "predicate: incidental detail (product image) → URL",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=target_url,
            target_variant_id=target.url_hash,
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        # IC has no explicit "anchor" — the user incidentally noticed a detail
        # while browsing. For multi-session retrieval contracts, treat the
        # incidentally-marked page (the target) as the carry-forward anchor:
        # that's the thing a later session would refer back to.
        memory_contract=MemoryContract(
            encodes=[target.url_hash],
            retrieves_from_prior=[],
            must_carry_into_next=[target.url_hash],
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )
