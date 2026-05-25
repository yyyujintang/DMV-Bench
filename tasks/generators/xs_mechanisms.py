"""Cross-session mechanism generators (SA_xs / IC_xs / VL_xs / PD_xs).

All four share the coord_3 structure:
    Session 0: Setup A — show anchor_0 in cat_0
    Session 1: Setup B — show anchor_1 in cat_1  (sub_task = Setup)
    Session 2: <mechanism>_xs recall — find an item in cat_2 whose varKey
               is determined by a memory-dependent rule.

Each mechanism encodes a different memory operation; the GT formula is
strict (cumulative dependency on prior anchors). NoMemory baseline is
expected to be at ~25% chance; visual-memory baselines (CoMEM, DualChannel,
HYMEM) should beat that meaningfully.

Common structure:
    - Each mechanism picks varKeys for anchor_0 and anchor_1 from {var_a..d}
    - The recall session has its own user message phrased per mechanism
    - GT in cat_2 = variant whose varKey matches the mechanism's rule

| Mech     | anchor_0.varKey, anchor_1.varKey | recall GT varKey rule         |
|----------|----------------------------------|------------------------------|
| SA_xs    | K, K (same)                       | shared K                     |
| IC_xs    | K0, K1 (different)                | K0 (the FIRST anchor)        |
| VL_xs    | K0, K1 (different)                | K1 (the SECOND anchor)        |
| PD_xs    | K0 then K1 — user "changes mind"  | K1 (current preference)      |

The user message for the recall session embeds the mechanism's reference
explicitly ("the FIRST item", "the SECOND collection", "your current
preference"), so the agent must retrieve the appropriate prior anchor
and read its visual style.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Tuple

from ._common import (
    ALL_STYLES,
    VAR_KEY_TO_STYLE,
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
from ..schema.multisession import MultiSessionTask, SessionRef, save_multi_session


GENERATOR_VERSION = "xs_mechanisms.v1"

# Sequence categories we draw from (avoid ones with very thin variant pools)
DEFAULT_CAT_SEQUENCE = ["rugs", "chairs", "lamps", "vases", "cushions", "tables"]


def _variant_at(catalogue: VariantCatalogue, cat: str, tier: int, var_key: str) -> Variant | None:
    for v in catalogue.variants_in(cat):
        if v.grain_tier == tier and v.var_key == var_key:
            return v
    return None


# ---------------------------------------------------------------------------
# Sub-task TaskInstance builders
# ---------------------------------------------------------------------------

def _setup_subtask(anchor: Variant, seed: int, sequence_idx: int,
                   intent_text: str) -> TaskInstance:
    """A Setup session: show one anchor; agent navigates to it. Always
    trivially correct (the runner short-circuits Setup-only)."""
    noun = coarse_noun(anchor.category_slug)
    anchor_url = f"/product/{anchor.url_hash}"
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content=intent_text,
             references_variant=anchor.url_hash),
        Turn(turn_index=2, role="agent", mode="recall",
             expected_actions=[AgentAction(action_type="navigate", target_url=anchor_url)],
             expected_url=anchor_url,
             recall_turn_metadata=RecallTurnMetadata(
                 anchor_invisibility_targets=[], expected_memory_usage="positive",
             )),
    ]
    task_id = (f"setup{sequence_idx}_{anchor.category_slug}_{anchor.var_key}"
               f"_t{anchor.grain_tier}_s{seed:04d}")
    return TaskInstance(
        task_id=task_id, sub_task="Setup",
        grain_tier=anchor.grain_tier, category_ids=[anchor.category_slug],
        variants_used=[anchor.url_hash], turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[anchor.url_hash],
            recall_turn_indices=[2], memory_gated_branch_turn=2,
            cross_turn_predicates=[f"setup#{sequence_idx} exposure"],
        ),
        ground_truth=GroundTruth(final_action="navigate", target_url=anchor_url,
                                   target_variant_id=anchor.url_hash),
        success_criteria=SuccessCriteria(type="url_match", evaluator_fn="match_final_url"),
        memory_contract=MemoryContract(
            encodes=[anchor.url_hash], retrieves_from_prior=[],
            must_carry_into_next=[anchor.url_hash],
        ),
        metadata=TaskMetadata(generated_by="llm", generator_seed=seed,
                                generator_version=GENERATOR_VERSION),
    )


def _recall_subtask(
    mechanism: str,
    sub_task: str,
    user_msg: str,
    new_cat: str,
    gt: Variant,
    required_prior_anchors: list[str],
    seed: int,
    cross_predicates: list[str],
) -> TaskInstance:
    """The XS recall session that demands cross-session memory."""
    gt_url = f"/product/{gt.url_hash}"
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding", content=user_msg),
        Turn(turn_index=2, role="agent", mode="recall",
             expected_actions=[AgentAction(action_type="navigate", target_url=gt_url)],
             expected_url=gt_url,
             recall_turn_metadata=RecallTurnMetadata(
                 anchor_invisibility_targets=["page_grid", "recently_viewed"],
                 expected_memory_usage="positive",
             )),
    ]
    task_id = f"{mechanism.lower()}_to_{new_cat}_s{seed:04d}"
    return TaskInstance(
        task_id=task_id, sub_task=sub_task,
        grain_tier=gt.grain_tier, category_ids=[new_cat],
        variants_used=[gt.url_hash], turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[gt.url_hash],
            recall_turn_indices=[2], memory_gated_branch_turn=2,
            cross_turn_predicates=cross_predicates,
        ),
        ground_truth=GroundTruth(final_action="navigate", target_url=gt_url,
                                   target_variant_id=gt.url_hash),
        success_criteria=SuccessCriteria(type="url_match", evaluator_fn="match_final_url"),
        memory_contract=MemoryContract(
            encodes=[gt.url_hash], retrieves_from_prior=required_prior_anchors,
            must_carry_into_next=required_prior_anchors + [gt.url_hash],
        ),
        metadata=TaskMetadata(generated_by="llm", generator_seed=seed,
                                generator_version=GENERATOR_VERSION),
    )


# ---------------------------------------------------------------------------
# Mechanism-specific user messages (memory-required cues)
# ---------------------------------------------------------------------------

def _sa_xs_msg(noun_0: str, noun_1: str, new_noun: str) -> str:
    return (
        f"I showed you two items earlier — a {noun_0} and a {noun_1}. "
        f"Look at both of them in your memory; they share a common overall style. "
        f"Find a {new_noun} that fits that same shared style closely."
    )


def _ic_xs_msg(noun_0: str, noun_1: str, new_noun: str) -> str:
    return (
        f"Earlier I showed you two items: first a {noun_0}, then a {noun_1}. "
        f"Now please look back at the very FIRST one — the {noun_0} — and find "
        f"a {new_noun} whose style matches that FIRST item specifically. Ignore "
        f"the second item's style for this one."
    )


def _vl_xs_msg(noun_0: str, noun_1: str, new_noun: str) -> str:
    # VL_xs: visual landmark / palette recall. Uses ordinal "second" but
    # framed around the VISUAL palette of that anchor, distinct from
    # IC_xs's "match the first item's style" framing. The agent must
    # retrieve anchor_1's image and read its palette, not its style label.
    return (
        f"Earlier I showed you a {noun_0} and then a {noun_1}. Now I'm focused "
        f"on the palette of the SECOND one — the {noun_1}. Find a {new_noun} "
        f"whose overall colour palette and tones match that {noun_1}'s palette."
    )


def _pd_xs_msg(noun_0: str, noun_1: str, new_noun: str) -> str:
    return (
        f"Earlier I showed you a {noun_0} and then changed my mind in favour of "
        f"a {noun_1} instead — that's my current direction. Find a {new_noun} "
        f"that matches my CURRENT direction (the {noun_1}'s style)."
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

Mechanism = Literal["SA_xs", "IC_xs", "VL_xs", "PD_xs"]


def generate_coord3_xs(
    mechanism: Mechanism,
    catalogue: VariantCatalogue,
    seed: int,
    tier: int = 1,
    cat_sequence: list[str] | None = None,
    val_pool_root: Path | None = None,
    multi_pool_root: Path | None = None,
) -> Tuple[list[TaskInstance], MultiSessionTask]:
    """Build one coord_3 cross-session task for the given mechanism.

    Returns (3 sub-task TaskInstances, parent MultiSessionTask).
    """
    rng_local = random.Random(f"xs.{mechanism}.{seed}")
    cats = cat_sequence or DEFAULT_CAT_SEQUENCE
    cats_with_variants = [c for c in cats if catalogue.variants_in(c)]
    if len(cats_with_variants) < 3:
        raise RuntimeError(f"need ≥3 categories; got {cats_with_variants}")
    cat_0, cat_1, cat_2 = rng_local.sample(cats_with_variants, 3)

    # Pick varKeys per mechanism
    var_keys = ["var_a", "var_b", "var_c", "var_d"]
    if mechanism == "SA_xs":
        # Both anchors share the SAME varKey
        K = rng_local.choice(var_keys)
        K0, K1 = K, K
        gt_varkey = K
    elif mechanism == "IC_xs":
        # Different varKeys; GT matches the FIRST
        K0, K1 = rng_local.sample(var_keys, 2)
        gt_varkey = K0
    elif mechanism == "VL_xs":
        # Different varKeys; GT matches the SECOND
        K0, K1 = rng_local.sample(var_keys, 2)
        gt_varkey = K1
    elif mechanism == "PD_xs":
        # Different varKeys; user "changes mind" to second → GT matches SECOND
        K0, K1 = rng_local.sample(var_keys, 2)
        gt_varkey = K1
    else:
        raise ValueError(f"unknown mechanism: {mechanism!r}")

    anchor_0 = _variant_at(catalogue, cat_0, tier, K0)
    anchor_1 = _variant_at(catalogue, cat_1, tier, K1)
    gt = _variant_at(catalogue, cat_2, tier, gt_varkey)
    if anchor_0 is None or anchor_1 is None or gt is None:
        raise RuntimeError(
            f"missing variant: cat_0={cat_0}/{K0}, cat_1={cat_1}/{K1}, "
            f"cat_2={cat_2}/{gt_varkey} at tier {tier}"
        )

    noun_0 = coarse_noun(cat_0)
    noun_1 = coarse_noun(cat_1)
    noun_2 = coarse_noun(cat_2)

    # Setup A — show anchor_0
    setup_0 = _setup_subtask(
        anchor_0, seed, sequence_idx=0,
        intent_text=(
            f"Here's a {noun_0} I want to consider for my home — I'd like to "
            f"build a coordinated look around it. Keep this style in mind."
        ),
    )
    # Setup B — show anchor_1 with a NEUTRAL intent (no recency/preference
    # bias). Critically, the setup phrase must NOT hint that this is "the
    # one that matters" — otherwise even NoMemory baselines can guess by
    # defaulting to "the latest thing in conversation". Each mechanism's
    # recall session is what disambiguates which prior anchor to use.
    if mechanism == "PD_xs":
        # PD specifically NEEDS a preference-shift hint in setup_B to
        # establish the drift narrative (that's the core of the mechanism).
        # The shift is between two items both presented neutrally; the
        # SHIFT itself is the signal, not "this one is special".
        setup_1_intent = (
            f"Actually, on second thought I've been looking at this {noun_1} "
            f"instead. I think I'd rather go in this direction now."
        )
    elif mechanism == "SA_xs":
        setup_1_intent = (
            f"And here's another item I'm also considering — a {noun_1}."
        )
    elif mechanism == "VL_xs":
        setup_1_intent = (
            f"Here's another piece I've been browsing — a {noun_1}."
        )
    elif mechanism == "IC_xs":
        setup_1_intent = (
            f"And here's a {noun_1} I also looked at."
        )
    setup_1 = _setup_subtask(anchor_1, seed, sequence_idx=1, intent_text=setup_1_intent)

    # Recall session: mechanism-specific user message + GT
    if mechanism == "SA_xs":
        user_msg = _sa_xs_msg(noun_0, noun_1, noun_2)
    elif mechanism == "IC_xs":
        user_msg = _ic_xs_msg(noun_0, noun_1, noun_2)
    elif mechanism == "VL_xs":
        user_msg = _vl_xs_msg(noun_0, noun_1, noun_2)
    elif mechanism == "PD_xs":
        user_msg = _pd_xs_msg(noun_0, noun_1, noun_2)

    required = [anchor_0.url_hash, anchor_1.url_hash]
    cross_predicates = [
        f"requires-memory-of: {anchor_0.url_hash[:8]} and {anchor_1.url_hash[:8]}",
        f"mechanism: {mechanism}",
        f"GT category: {cat_2}",
    ]
    recall = _recall_subtask(
        mechanism=mechanism, sub_task=mechanism, user_msg=user_msg,
        new_cat=cat_2, gt=gt, required_prior_anchors=required,
        seed=seed, cross_predicates=cross_predicates,
    )

    # Persist to validated pool
    val_root = val_pool_root or (Path(__file__).resolve().parents[2]
                                  / "tasks" / "pool" / "validated")
    for inst in [setup_0, setup_1, recall]:
        dst = val_root / inst.sub_task / f"{inst.task_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(inst.model_dump_json(indent=2))

    # Compose MultiSessionTask
    refs = [
        SessionRef(
            task_id=setup_0.task_id, sub_task="Setup", order_index=0,
            retrieves_from_prior_override=[],
            must_carry_override=[anchor_0.url_hash],
            cross_session_user_preamble="",
        ),
        SessionRef(
            task_id=setup_1.task_id, sub_task="Setup", order_index=1,
            retrieves_from_prior_override=[anchor_0.url_hash],
            must_carry_override=[anchor_0.url_hash, anchor_1.url_hash],
            cross_session_user_preamble="",
        ),
        SessionRef(
            task_id=recall.task_id, sub_task=mechanism, order_index=2,
            retrieves_from_prior_override=required,
            must_carry_override=required + [gt.url_hash],
            cross_session_user_preamble="",
        ),
    ]
    ms_id = f"coord3_{mechanism.lower()}_{cat_0}_{cat_1}_to_{cat_2}_t{tier}_s{seed:04d}"
    multi = MultiSessionTask(
        task_id=ms_id, variant=f"coord_3_{mechanism.lower()}",
        sessions=refs, cumulative_turn_budget=60,
        cumulative_gt=[gt.url_hash],
        homogeneous=False,
        metadata={
            # Public (already exposed via user prompt / nav):
            "anchor_0_url_hash": anchor_0.url_hash,
            "anchor_1_url_hash": anchor_1.url_hash,
            "cat_sequence": [cat_0, cat_1, cat_2],
            "tier": tier,
            "generator": GENERATOR_VERSION,
            "mechanism": mechanism,
            # GT-leak-safe: do NOT include varKey or style label.
        },
    )

    multi_root = multi_pool_root or (Path(__file__).resolve().parents[2]
                                       / "tasks" / "pool" / "multisession_xs")
    save_multi_session(multi, pool=multi_root)
    return [setup_0, setup_1, recall], multi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mechanism", choices=["SA_xs", "IC_xs", "VL_xs", "PD_xs", "all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tier", type=int, default=1)
    args = p.parse_args()
    cat = VariantCatalogue()
    mechs = ["SA_xs", "IC_xs", "VL_xs", "PD_xs"] if args.mechanism == "all" else [args.mechanism]
    for m in mechs:
        try:
            _, multi = generate_coord3_xs(m, cat, seed=args.seed, tier=args.tier)
            print(f"[ok] {multi.task_id}")
        except Exception as e:
            print(f"[skip] {m}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    _cli()
