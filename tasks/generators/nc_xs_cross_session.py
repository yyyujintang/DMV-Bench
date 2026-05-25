"""NC_xs — Negative Constraint, cross-session variant.

Coordinated-set shopping: a customer assembles a visually coordinated set
across 3+ sessions. The Setup session shows an anchor item; each
NC_xs session asks for the next item in a different category that
**matches the anchor's varKey** while excluding a stated style.

GT formula (strict, memory-dependent):

    GT_i = the unique variant V where:
        V.category   = new_category_i
        V.grain_tier = tier_0
        V.var_key    = anchor_0.var_key       ← only retrievable from memory
        V.style_slug != rejected_style_i      ← stated in user prompt

Each (category, tier) pool has exactly four variants (one per varKey), so
the matching varKey uniquely identifies GT. The rejected_style is picked
≠ anchor.style_slug, making the constraint non-trivial but satisfiable.

A NoMemory agent has no access to anchor_0; conditional on style ≠
rejected_style, three styles remain, so chance is ~33% per session.

Generator output:
    - 3 TaskInstance JSONs (Setup + 2 NC_xs), saved to
      `tasks/pool/validated/<sub_task>/` so the runner finds them
    - 1 MultiSessionTask referencing them, saved to
      `tasks/pool/multisession_xs/`
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Tuple

from ._common import (
    ALL_STYLES,
    STYLE_TO_VAR_KEY,
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

GENERATOR_VERSION = "nc_xs.v1"

# Order of categories to draw the coord-3 set from (rug → chair → lamp →
# vase → cushion → table). The Setup is the FIRST entry; subsequent
# NC_xs sessions use the next entries in order.
DEFAULT_CAT_SEQUENCE = ["rugs", "chairs", "lamps", "vases", "cushions", "tables"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variant_at(catalogue: VariantCatalogue, cat: str, tier: int,
                var_key: str) -> Variant | None:
    """Return the variant in (cat, tier) with the given var_key, or None."""
    for v in catalogue.variants_in(cat):
        if v.grain_tier == tier and v.var_key == var_key:
            return v
    return None


def _build_setup_subtask(anchor: Variant, seed: int) -> TaskInstance:
    """Setup session: customer shows the anchor; agent visits its detail page.
    Marked is_setup_only via metadata so the runner skips free-form."""
    noun = coarse_noun(anchor.category_slug)
    anchor_url = f"/product/{anchor.url_hash}"
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content=f"Here's a {noun} I really like. I'd like to build a set "
                     f"around it — keep this one in mind for what comes next.",
             references_variant=anchor.url_hash),
        Turn(turn_index=2, role="agent", mode="recall",
             expected_actions=[AgentAction(action_type="navigate",
                                            target_url=anchor_url)],
             expected_url=anchor_url,
             recall_turn_metadata=RecallTurnMetadata(
                 anchor_invisibility_targets=[],
                 expected_memory_usage="positive",
             )),
    ]
    task_id = f"setup_{anchor.category_slug}_{anchor.var_key}_t{anchor.grain_tier}_s{seed:04d}"
    return TaskInstance(
        task_id=task_id,
        sub_task="Setup",
        grain_tier=anchor.grain_tier,
        category_ids=[anchor.category_slug],
        variants_used=[anchor.url_hash],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[anchor.url_hash],
            recall_turn_indices=[2],
            memory_gated_branch_turn=2,
            cross_turn_predicates=["setup: anchor exposure for downstream sessions"],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=anchor_url,
            target_variant_id=anchor.url_hash,
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        memory_contract=MemoryContract(
            encodes=[anchor.url_hash],
            retrieves_from_prior=[],
            must_carry_into_next=[anchor.url_hash],
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )


def _build_nc_xs_subtask(
    anchor: Variant,
    new_cat: str,
    gt: Variant,
    rejected_style: str,
    seed: int,
    session_idx: int,
) -> TaskInstance:
    """One NC_xs session referencing the prior Setup anchor.

    The user prompt:
      1. Explicitly references "the {anchor_noun} I showed earlier".
      2. Names the rejected style.
      3. Does NOT mention the anchor's varKey or style — agent must recall.

    GT = `gt` (unique variant in new_cat with same var_key as anchor, with
    style != rejected_style).
    """
    anchor_noun = coarse_noun(anchor.category_slug)
    new_noun = coarse_noun(new_cat)
    gt_url = f"/product/{gt.url_hash}"
    user_msg = (
        f"Remember the {anchor_noun} I showed you earlier? I want to find a "
        f"{new_noun} that fits the same overall style and tone as that "
        f"{anchor_noun} — I'm building a coordinated set. Please find one that "
        f"matches its style closely, but not the {rejected_style} version."
    )
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding", content=user_msg),
        Turn(turn_index=2, role="agent", mode="recall",
             expected_actions=[AgentAction(action_type="navigate",
                                            target_url=gt_url)],
             expected_url=gt_url,
             recall_turn_metadata=RecallTurnMetadata(
                 anchor_invisibility_targets=["page_grid", "recently_viewed"],
                 expected_memory_usage="positive",
             )),
    ]
    task_id = (f"nc_xs_{anchor.category_slug}_to_{new_cat}_"
               f"{anchor.var_key}_t{anchor.grain_tier}_"
               f"reject_{rejected_style}_s{seed:04d}_i{session_idx}")
    return TaskInstance(
        task_id=task_id,
        sub_task="NC_xs",
        grain_tier=anchor.grain_tier,
        category_ids=[new_cat],
        variants_used=[gt.url_hash],
        turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[gt.url_hash],
            recall_turn_indices=[2],
            memory_gated_branch_turn=2,
            # GT-leak-safe predicates: do NOT include varKey or style label.
            # The agent must derive the matching style from the anchor's
            # IMAGE in memory, not from a task-level annotation.
            cross_turn_predicates=[
                f"requires-memory-of: {anchor.url_hash[:8]}",
                f"style != {rejected_style}",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate",
            target_url=gt_url,
            target_variant_id=gt.url_hash,
        ),
        success_criteria=SuccessCriteria(
            type="url_match",
            evaluator_fn="match_final_url",
        ),
        memory_contract=MemoryContract(
            encodes=[gt.url_hash],
            retrieves_from_prior=[anchor.url_hash],
            must_carry_into_next=[anchor.url_hash, gt.url_hash],
        ),
        metadata=TaskMetadata(
            generated_by="llm",
            generator_seed=seed,
            generator_version=GENERATOR_VERSION,
        ),
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_coord3_nc_xs(
    catalogue: VariantCatalogue,
    seed: int,
    tier: int = 1,
    cat_sequence: list[str] | None = None,
    val_pool_root: Path | None = None,
    multi_pool_root: Path | None = None,
) -> Tuple[list[TaskInstance], MultiSessionTask]:
    """Build one coord_3 task: Setup → NC_xs (cat_1) → NC_xs (cat_2).

    Writes the 3 sub-task JSONs into `val_pool_root` (default
    `tasks/pool/validated/<sub>/`) and the multi-session JSON into
    `multi_pool_root` (default `tasks/pool/multisession_xs/`).
    """
    rng = seeded_rng(seed, f"nc_xs.coord3")
    cats = cat_sequence or DEFAULT_CAT_SEQUENCE
    # Need at least 3 categories with variants at the chosen tier.
    cats_with_variants = [c for c in cats if catalogue.variants_in(c)]
    if len(cats_with_variants) < 3:
        raise RuntimeError(f"need ≥3 categories with variants; got {cats_with_variants}")
    # Pick a deterministic order (anchor + 2 followups) from cats_with_variants.
    rng_local = random.Random(f"nc_xs.cats.{seed}")
    chosen_cats = rng_local.sample(cats_with_variants, 3)
    cat_0, cat_1, cat_2 = chosen_cats

    # Pick a varKey uniformly at random.
    chosen_var_key = rng_local.choice(["var_a", "var_b", "var_c", "var_d"])
    anchor_style = VAR_KEY_TO_STYLE[chosen_var_key]

    anchor = _variant_at(catalogue, cat_0, tier, chosen_var_key)
    if anchor is None:
        raise RuntimeError(f"no anchor in ({cat_0}, t{tier}, {chosen_var_key})")
    # Rejected style for each NC_xs session: pick a style ≠ anchor.style.
    # Different rejected style per session for variety.
    other_styles = [s for s in ALL_STYLES if s != anchor_style]
    rng_local.shuffle(other_styles)
    rejected_1, rejected_2 = other_styles[0], other_styles[1 % len(other_styles)]

    gt_1 = _variant_at(catalogue, cat_1, tier, chosen_var_key)
    gt_2 = _variant_at(catalogue, cat_2, tier, chosen_var_key)
    if gt_1 is None or gt_2 is None:
        raise RuntimeError(f"no GT at varKey={chosen_var_key} in "
                           f"({cat_1}/{cat_2}, t{tier})")

    setup_task = _build_setup_subtask(anchor, seed)
    nc_xs_1 = _build_nc_xs_subtask(anchor, cat_1, gt_1, rejected_1, seed, session_idx=1)
    nc_xs_2 = _build_nc_xs_subtask(anchor, cat_2, gt_2, rejected_2, seed, session_idx=2)

    # Persist sub-task instances under the validated pool so the runner
    # finds them via standard load_session_instance.
    val_root = val_pool_root or (Path(__file__).resolve().parents[2]
                                 / "tasks" / "pool" / "validated")
    for inst in [setup_task, nc_xs_1, nc_xs_2]:
        dst = val_root / inst.sub_task / f"{inst.task_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(inst.model_dump_json(indent=2))

    # Compose the MultiSessionTask. retrieves_from_prior is the anchor for
    # session 1, the anchor+gt_1 for session 2.
    refs = [
        SessionRef(
            task_id=setup_task.task_id, sub_task="Setup", order_index=0,
            retrieves_from_prior_override=[],
            must_carry_override=[anchor.url_hash],
            cross_session_user_preamble="",
        ),
        SessionRef(
            task_id=nc_xs_1.task_id, sub_task="NC_xs", order_index=1,
            retrieves_from_prior_override=[anchor.url_hash],
            must_carry_override=[anchor.url_hash, gt_1.url_hash],
            cross_session_user_preamble="",   # already embedded in user msg
        ),
        SessionRef(
            task_id=nc_xs_2.task_id, sub_task="NC_xs", order_index=2,
            retrieves_from_prior_override=[anchor.url_hash, gt_1.url_hash],
            must_carry_override=[anchor.url_hash, gt_1.url_hash, gt_2.url_hash],
            cross_session_user_preamble="",
        ),
    ]
    ms_id = f"coord3_nc_xs_{cat_0}_{chosen_var_key}_t{tier}_s{seed:04d}"
    multi = MultiSessionTask(
        task_id=ms_id,
        variant="coord_3_nc_xs",
        sessions=refs,
        cumulative_turn_budget=20 * 3,  # rough upper-bound
        cumulative_gt=[gt_1.url_hash, gt_2.url_hash],
        homogeneous=False,
        metadata={
            # Public (visible to runner / agent indirectly via user prompt):
            "anchor_url_hash": anchor.url_hash,        # exposed via references_variant
            "anchor_category": cat_0,                   # exposed via "the X I showed earlier"
            "cat_sequence": [cat_0, cat_1, cat_2],     # user navigation path
            "rejected_styles": [rejected_1, rejected_2],  # in user prompt
            "tier": tier,                                # design parameter
            "generator": GENERATOR_VERSION,
            # GT-leak fix (paper rigor): anchor_var_key and anchor_style
            # were removed. They ARE the hidden GT — the agent must
            # derive them from anchor's image via memory, never from a
            # task-level label. Scoring relies on per-session
            # ground_truth.target_url which is set per session JSON.
        },
    )

    multi_root = (multi_pool_root
                  or Path(__file__).resolve().parents[2]
                  / "tasks" / "pool" / "multisession_xs")
    save_multi_session(multi, pool=multi_root)
    return [setup_task, nc_xs_1, nc_xs_2], multi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse, json
    p = argparse.ArgumentParser(description="Generate coord_3 NC_xs tasks")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tier", type=int, default=1)
    args = p.parse_args()
    cat = VariantCatalogue()
    for i in range(args.n):
        seed = args.seed + i
        try:
            subtasks, multi = generate_coord3_nc_xs(cat, seed=seed, tier=args.tier)
            print(f"[ok] {multi.task_id}  anchor={multi.metadata.get('anchor_url_hash')}  "
                  f"varKey={multi.metadata.get('anchor_var_key')}")
        except Exception as e:
            print(f"[skip seed={seed}] {type(e).__name__}: {e}")


if __name__ == "__main__":
    _cli()
