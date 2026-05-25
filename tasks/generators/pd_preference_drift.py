"""PD — Preference Drift (proposal_tasks_v2.md §7).

W7v2 / Phase-4.5 implementation. 13-turn long-horizon generator with
explicit Pattern 1/2/3 revision grammar, distractor pool, and CLIP-
centroid ground-truth resolver.

Structure:
  Phase 1 (T1–T3): initial setup, 2 preferences declared, 1 positive
                    reaction.
  Phase 2 (T4–T7): mid-task elaboration, 1–2 extra preferences, 1
                    Pattern-3 revision.
  Phase 3 (T8–T12): distractor + drift — at least one off-topic
                    distractor, one aborted exploration, one
                    self-contradiction-and-withdrawal, plus 1–2 more
                    Pattern-1/2 revisions.
  Phase 4 (T13): probe — "Pick the one product that fits everything I
                    still want."

PD-Easy: 3 preferences, 2 total revisions (1 P2 + 1 P3).
PD-Hard: 5 preferences, 4 revisions, at least one Pattern-2 retiring
         a Phase-1 preference (maximum temporal distance).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from ._common import ALL_STYLES, VariantCatalogue, Variant, coarse_noun, seeded_rng
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
from ..schema.preference_state import (
    PATTERN1_TEMPLATES,
    PATTERN2_TEMPLATES,
    PATTERN3_TEMPLATES,
    PreferenceSlot,
    PreferenceState,
    apply_pattern1,
    apply_pattern2,
    apply_pattern3,
    final_active_state,
)
from ..scoring.pd_resolver import resolve_pd
from ..scoring.uniqueness import Ambiguous, _load_cache

GENERATOR_VERSION = "pd.v2"


DISTRACTOR_POOL: list[str] = [
    "By the way, what's your return policy?",
    "Quick question — how long does shipping usually take?",
    "Do you offer gift wrapping?",
    "Out of curiosity, how long has this brand been around?",
    "Do you have a price-match guarantee?",
    "Any plans to add international shipping?",
    "What's the standard warranty?",
    "Is there a way to expedite delivery?",
    "Do you have any seasonal sales coming up?",
    "I might gift this — can you ship to a different address?",
    "How do I track an order once it's placed?",
    "Are there any current discount codes I should know about?",
    "Just wondering — where are these typically made?",
    "Do you have a physical store or just online?",
    "Can I return something after 30 days if needed?",
]


# Style-name avoidance: PD chat does NOT say "vintage" / "modern" /
# etc. in user lines. Instead the user uses descriptive copy ("warm
# retro look", "clean geometric look") that maps to one style.
STYLE_DESCRIPTOR = {
    "modern":      "a clean modern look",
    "minimalist":  "an understated minimalist look",
    "vintage":     "a warm vintage look",
    "industrial":  "an industrial loft look",
}


def _bare(desc: str) -> str:
    """Strip the leading 'a '/'an ' article so the descriptor can be
    interpolated into slots that already supply a determiner like
    'the ___ idea'. Without this we end up with 'the a clean modern
    look idea' (double article)."""
    for art in ("a ", "an "):
        if desc.lower().startswith(art):
            return desc[len(art):]
    return desc

COLOR_DESCRIPTORS = {
    # Color name → casual descriptor used in chat lines.
    "Tobacco":   "a warm tobacco shade",
    "Burgundy":  "a deep burgundy",
    "Plum":      "a moody plum",
    "Caramel":   "a soft caramel",
    "Sage":      "a sage green",
    "Bone":      "a soft bone tone",
    "Charcoal":  "a charcoal grey",
    "Slate":     "a slate cool grey",
    "Olive":     "an olive green",
    "Ivory":     "an ivory off-white",
    "Dove":      "a dove grey",
    "Mist":      "a misty pale",
}


def _pick_initial_setup(
    catalogue: VariantCatalogue,
    rng: random.Random,
    cache: dict[str, list[float]],
) -> tuple[object, str, float, Variant, Variant]:
    """Choose (category, initial_style, initial_price_max, anchor_v1, anchor_v2)
    that establish Phase-1 state and have two products to browse."""
    cats = catalogue.categories.copy()
    rng.shuffle(cats)
    styles = list(ALL_STYLES)
    for cat in cats:
        for style in rng.sample(styles, len(styles)):
            pool = [v for v in catalogue.variants_in(cat.slug) if v.style_slug == style]
            if len(pool) < 2:
                continue
            v1, v2 = rng.sample(pool, 2)
            price_max = max(v1.price, v2.price) + 5.0
            return cat, style, price_max, v1, v2
    raise RuntimeError("PD: no feasible initial setup")


def _maybe_revise_style(
    state: PreferenceState,
    rng: random.Random,
    new_style: str,
    turn_index: int,
) -> str:
    """Apply a Pattern-2 replacement on the style slot. Returns chat content."""
    slot = state["style"]
    old_value = slot.value
    tpl = rng.choice(PATTERN2_TEMPLATES)
    content = tpl.format(
        old=STYLE_DESCRIPTOR.get(old_value, old_value),
        new=STYLE_DESCRIPTOR.get(new_style, new_style),
    )
    apply_pattern2(state, _Pattern2Op(
        slot_type="style", old_value=old_value, new_value=new_style,
        turn_index=turn_index,
    ))
    return content


class _Pattern1Op:
    def __init__(self, slot_type, turn_index):
        self.slot_type, self.turn_index = slot_type, turn_index


class _Pattern2Op:
    def __init__(self, slot_type, old_value, new_value, turn_index):
        self.slot_type = slot_type
        self.old_value = old_value
        self.new_value = new_value
        self.turn_index = turn_index


class _Pattern3Op:
    def __init__(self, slot_type, new_value, turn_index):
        self.slot_type, self.new_value, self.turn_index = slot_type, new_value, turn_index


def generate(
    catalogue: VariantCatalogue,
    seed: int,
    difficulty: Literal["easy", "hard"] = "easy",
) -> TaskInstance:
    rng = seeded_rng(seed, f"pd.{difficulty}")
    cache = _load_cache()

    # The PD trajectory is a coupling of (initial_style, final_style,
    # final_price_max). We retry up to 30 times to find a (cat, anchors)
    # whose resulting hard-filter leaves a unique CLIP-resolved GT.
    last_err: str | None = None
    for attempt in range(30):
        try:
            return _attempt_generate(catalogue, rng, cache, difficulty, seed, attempt)
        except (Ambiguous, RuntimeError, ValueError) as e:
            last_err = f"attempt {attempt}: {type(e).__name__}: {e}"
            continue
    raise RuntimeError(f"PD seed {seed} {difficulty}: 30 retries exhausted; last={last_err}")


def _attempt_generate(
    catalogue: VariantCatalogue,
    rng: random.Random,
    cache: dict[str, list[float]],
    difficulty: Literal["easy", "hard"],
    seed: int,
    attempt: int,
) -> TaskInstance:
    cat, style_initial, price_initial, anchor1, anchor2 = _pick_initial_setup(
        catalogue, rng, cache
    )
    noun = coarse_noun(cat.slug)

    state: PreferenceState = {}
    positive_reactions: list[str] = []
    turns: list[Turn] = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
    ]

    # ---------- Phase 1: T1–T3 ----------
    # T1 — initial preferences (style + price_max)
    t1_content = (
        f"I'm looking for a {noun}. I want {STYLE_DESCRIPTOR[style_initial]}, "
        f"under ${price_initial:.0f}."
    )
    turns.append(Turn(turn_index=1, role="user", mode="encoding", content=t1_content))
    state["style"] = PreferenceSlot(
        slot_type="style", value=style_initial, status="active",
        source_turn=1, history=[],
    )
    state["price_max"] = PreferenceSlot(
        slot_type="price_max", value=price_initial, status="active",
        source_turn=1, history=[],
    )
    state["category"] = PreferenceSlot(
        slot_type="category", value=cat.slug, status="active",
        source_turn=1, history=[],
    )
    # T2 — agent navigates
    turns.append(Turn(
        turn_index=2, role="agent", mode="encoding",
        expected_actions=[AgentAction(action_type="navigate",
                                       target_url=f"/collection/{anchor1.collection_slug}")],
        expected_url=f"/collection/{anchor1.collection_slug}",
    ))
    # T3 — positive reaction on anchor1
    turns.append(Turn(
        turn_index=3, role="user", mode="encoding",
        content=f"I like that {_bare(COLOR_DESCRIPTORS.get(anchor1.color_name, anchor1.color_name.lower()))} one. Show me more in similar tones.",
        references_variant=anchor1.url_hash,
    ))
    positive_reactions.append(anchor1.url_hash)

    # ---------- Phase 2: T4–T7 ----------
    # T4 — add a colour preference (color of anchor1)
    color_pref = anchor1.color_name
    turns.append(Turn(
        turn_index=4, role="user", mode="encoding",
        content=f"By the way, nothing too sleek — I'd prefer something cozy.",
    ))
    state["ornament"] = PreferenceSlot(
        slot_type="ornament", value="cozy", status="active",
        source_turn=4, history=[],
    )
    # T5 — agent shows anchor2
    turns.append(Turn(
        turn_index=5, role="agent", mode="encoding",
        expected_actions=[AgentAction(action_type="navigate",
                                       target_url=f"/product/{anchor2.url_hash}")],
        expected_url=f"/product/{anchor2.url_hash}",
    ))
    positive_reactions.append(anchor2.url_hash)
    # T6 — extra preference: material text-filter
    material_pref = "fabric"
    turns.append(Turn(
        turn_index=6, role="user", mode="encoding",
        content=f"Yes, much better. Also, I prefer {material_pref} over leather.",
    ))
    state["material"] = PreferenceSlot(
        slot_type="material", value=material_pref, status="active",
        source_turn=6, history=[],
    )
    # T7 — Pattern-3 revision: narrow the budget by 5–15% so the resolver
    # still has ≥ 2 candidates in (cat, style) after filtering.
    narrow_pct = rng.uniform(0.05, 0.15)
    new_price = round(price_initial * (1.0 - narrow_pct))
    if new_price <= anchor2.price:
        # Floor: keep both anchors visible in price filter
        new_price = round(max(anchor1.price, anchor2.price) + 1)
    pattern3_content = rng.choice(PATTERN3_TEMPLATES).format(
        old=f"under ${price_initial:.0f}", new=f"under ${new_price:.0f}", slot="budget",
    )
    turns.append(Turn(turn_index=7, role="user", mode="encoding", content=pattern3_content))
    apply_pattern3(state, _Pattern3Op("price_max", new_price, 7))

    # ---------- Phase 3: T8–T12 ----------
    # T8 — distractor: off-topic chat
    t8 = rng.choice(DISTRACTOR_POOL)
    turns.append(Turn(turn_index=8, role="user", mode="encoding", content=t8))
    # T9 — agent answers (distractor-handling)
    turns.append(Turn(
        turn_index=9, role="agent", mode="encoding",
        content="Sure, let me share the policy details with you.",
    ))
    # T10 — aborted exploration: user briefly explores another style
    other_style = next(s for s in ALL_STYLES if s != state["style"].value)
    turns.append(Turn(
        turn_index=10, role="user", mode="encoding",
        content=f"Do you have anything with {STYLE_DESCRIPTOR[other_style]}? Just curious.",
    ))
    turns.append(Turn(
        turn_index=11, role="agent", mode="encoding",
        expected_actions=[AgentAction(action_type="navigate",
                                       target_url=f"/category/{cat.slug}")],
        expected_url=f"/category/{cat.slug}",
    ))
    # T12 — pick a final_style that, combined with current price_max +
    # category, has ≥ 3 candidates in the catalog (so the resolver has
    # something to score and an ε margin chance).
    candidate_final_styles: list[str]
    current_price_max = state["price_max"].value
    if difficulty == "hard":
        candidate_final_styles = [
            s for s in ALL_STYLES
            if s != state["style"].value
            and sum(
                1 for v in catalogue.variants_in(cat.slug)
                if v.style_slug == s and v.price <= current_price_max
            ) >= 3
        ]
    else:
        candidate_final_styles = (
            [state["style"].value]
            if sum(
                1 for v in catalogue.variants_in(cat.slug)
                if v.style_slug == state["style"].value
                and v.price <= current_price_max
            ) >= 3
            else []
        )
    if not candidate_final_styles:
        raise RuntimeError(
            f"no viable final_style for (cat={cat.slug}, "
            f"price_max={current_price_max})"
        )
    final_style = rng.choice(candidate_final_styles)
    other_bare = _bare(STYLE_DESCRIPTOR[other_style])
    if difficulty == "hard":
        revise_content = _maybe_revise_style(state, rng, final_style, 12)
        t12_content = (
            f"Hmm, no, on second thought, ignore the {other_bare} idea. "
            + revise_content
        )
    else:
        # Easy: keep initial style; abort the exploration + 1 minor Pattern-1 revoke
        revoke_target = "ornament"
        if state.get(revoke_target) and state[revoke_target].status == "active":
            tpl = rng.choice(PATTERN1_TEMPLATES)
            t12_content = (
                f"Actually, ignore the {other_bare} idea. "
                + tpl.format(slot="cozy preference")
            )
            apply_pattern1(state, _Pattern1Op(revoke_target, 12))
        else:
            t12_content = f"Forget the {other_bare} idea, I'll stick with what I had."
    turns.append(Turn(turn_index=12, role="user", mode="encoding", content=t12_content))

    # ---------- Phase 4: T13 probe ----------
    turns.append(Turn(
        turn_index=13, role="user", mode="recall",
        content="Pick the one product from your store that fits everything I still want.",
    ))

    # ---------- Resolve ground truth ----------
    p_final = final_active_state(state)
    # Drop ornament + material from the strict resolver — those are loose
    # textual hints that often don't match the catalog vocabulary cleanly.
    # The centroid + style + price + category still narrow heavily; ε=0.05
    # decides among the survivors via visual similarity to anchors.
    p_filter = {k: v for k, v in p_final.items() if k not in ("ornament", "material")}
    result = resolve_pd(
        catalogue, p_filter, positive_reactions, cache,
    )
    gt_hash = result.url_hash

    turns.append(Turn(
        turn_index=14, role="agent", mode="recall",
        expected_actions=[AgentAction(action_type="navigate",
                                       target_url=f"/product/{gt_hash}")],
        expected_url=f"/product/{gt_hash}",
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "recently_viewed"],
            expected_memory_usage="positive",
        ),
    ))

    diff_tag = "h" if difficulty == "hard" else "e"
    return TaskInstance(
        task_id=f"pd_{cat.slug}_{diff_tag}_s{seed:04d}",
        sub_task="PD",
        grain_tier=anchor1.grain_tier,
        category_ids=[cat.slug],
        variants_used=[anchor1.url_hash, anchor2.url_hash, gt_hash],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[],
            recall_turn_indices=[14],
            memory_gated_branch_turn=14,
            cross_turn_predicates=[
                f"p_final.style: {p_filter.get('style', 'unset')}",
                f"p_final.price_max: {p_filter.get('price_max', 'unset')}",
                f"p_final.material: {p_filter.get('material', 'unset')}",
                f"positive_centroid: {len(positive_reactions)} reactions",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=f"/product/{gt_hash}",
            target_variant_id=gt_hash,
            accepted_alternatives=[],
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        # PD has no NCP anchor (the preference state IS the anchor), so the
        # auto-populator from anchor_variant_ids won't fire. Declare positive
        # reactions explicitly so multi-session composers can wire them
        # forward into later sessions' retrieves_from_prior.
        memory_contract=MemoryContract(
            encodes=[anchor1.url_hash, anchor2.url_hash],
            retrieves_from_prior=[],
            must_carry_into_next=[anchor1.url_hash, anchor2.url_hash, gt_hash],
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )
