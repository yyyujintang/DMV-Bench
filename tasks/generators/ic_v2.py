"""v2 IC-only task generator — coord_5 incidental-cue RELAY chain.

One task = 5 sessions. Pure visual-memory recall, scored by EXACT url_match.
No style judgement anywhere (styles overlap visually — see the smoke-3
post-mortem; art_deco vs vintage was indistinguishable to the VLM).

Each session browses BROWSE_PER_SESSION (=20) products spanning MULTIPLE
categories. One of the 20 is the "relay" product P_N whose incidental cue
X_N is recalled by the next session.

    S0 (Setup):  agent browses 20 cross-category products. P0 is the relay.
    S1 (browse 20 + recall X0): browse 20 (relay P1/X1), THEN the user asks
                 to go back to the EXACT product that had cue X0. GT = P0 url.
    S2 (browse 20 + recall X1): browse 20 (P2/X2); recall X1 → GT = P1.
    S3 (browse 20 + recall X2): browse 20 (P3/X3); recall X2 → GT = P2.
    S4 (browse 20 + recall X3): browse 20 (P4/X4, unused); recall X3 → GT = P3.

Why 20-product cross-category browse: the memory bank accumulates across all
5 sessions (20→40→60→80→100). With per-step retrieval at k=5, k≪bank — so
the retriever must genuinely SELECT the cued product from a large pool, not
trivially return one whole session (the relay-v1 flaw: 5-browse + k=5 was a
5-of-5 identity map with no retrieval pressure).

Why a relay (depth-1): session N's recall target P_{N-1} is the cue planted
during session N-1's browse, so session N strictly depends on session N-1.
The cue (color + object) is a GLOBALLY-UNIQUE identifier — v2 cue allocation
is bijective over all 1000 products — so naming "the {color} {object}" picks
exactly one product. The agent must remember WHICH product (URL) carried it.

GT is EXACT: success = final_url == /product/<P_{N-1}>. accepted_alternatives
is empty. A working visual memory retrieves the cued product's URL directly.

Cues are read from data/vismem_diag_v2/cue_registry.json — the authoritative
record of what was actually edited into each image.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

from ..schema.task_instance import (
    AgentAction, GroundTruth, MemoryContract, NcpMetadata, RecallTurnMetadata,
    SuccessCriteria, TaskInstance, TaskMetadata, Turn,
)
from ..schema.multisession import MultiSessionTask, SessionRef, save_multi_session


GENERATOR_VERSION = "ic_v2.coord5.relay.v2"

REPO_ROOT = Path(__file__).resolve().parents[2]
CUE_REGISTRY_PATH = REPO_ROOT / "data" / "vismem_diag_v2" / "cue_registry.json"
V2_VAL_POOL = REPO_ROOT / "tasks" / "pool_v2" / "validated"
V2_MULTI_POOL = REPO_ROOT / "tasks" / "pool_v2" / "multisession"

CATEGORIES = ["chair", "sofa", "lamp", "cushion", "vase",
              "rug", "table", "bookshelf", "plant_pot", "wall_art"]

# 20 products browsed per session, cross-category. 5 sessions => 100 distinct
# products per task (cues globally unique => within-task unique automatically).
BROWSE_PER_SESSION = 20
N_SESSIONS = 5


def _category_noun(cat: str) -> str:
    return {
        "chair": "chair", "sofa": "sofa", "lamp": "lamp", "cushion": "cushion",
        "vase": "vase", "rug": "rug", "table": "side table",
        "bookshelf": "bookshelf", "plant_pot": "plant pot",
        "wall_art": "wall art piece",
    }.get(cat, cat)


def _url(row: dict) -> str:
    return f"/product/{row['url_hash']}"


def _cue_phrase(row: dict) -> str:
    """Spoken description of the incidental cue — the recall key. The cue is
    NOT secret (unlike style); it is exactly what the user says they remember."""
    return f"{row['cue_color']} {row['cue_object']}"


# ---------------------------------------------------------------------------
# Per-session TaskInstance builders
# ---------------------------------------------------------------------------

def _categories_of(rows: list[dict]) -> list[str]:
    return sorted({r["cat"] for r in rows})


def _build_setup_browse(browse_rows: list[dict],
                        relay_row: dict, seed: int) -> TaskInstance:
    """Session 0 — Setup. Agent browses 20 cross-category products; the runner
    walks all turns (is_setup_only) and encodes each. Relay P0 lives here."""
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content="I'm furnishing a new place — let me browse around and "
                     "look at a mix of pieces to get a feel for what's on offer."),
    ]
    for i, row in enumerate(browse_rows):
        turns.append(Turn(
            turn_index=2 + i, role="agent", mode="encoding",
            expected_actions=[AgentAction(action_type="navigate", target_url=_url(row))],
            expected_url=_url(row),
        ))
    last_idx = 2 + len(browse_rows) - 1
    turns[-1] = turns[-1].model_copy(update={
        "mode": "recall",
        "recall_turn_metadata": RecallTurnMetadata(
            anchor_invisibility_targets=[], expected_memory_usage="positive"),
    })
    browsed = [r["url_hash"] for r in browse_rows]
    task_id = f"setup_browse_s{seed:04d}"
    return TaskInstance(
        task_id=task_id, sub_task="Setup", grain_tier=1,
        category_ids=_categories_of(browse_rows), variants_used=browsed, turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[relay_row["url_hash"]],
            recall_turn_indices=[last_idx], memory_gated_branch_turn=last_idx,
            cross_turn_predicates=[f"setup: browse {len(browsed)} cross-category products"],
        ),
        ground_truth=GroundTruth(
            final_action="navigate", target_url=_url(relay_row),
            target_variant_id=relay_row["url_hash"]),
        success_criteria=SuccessCriteria(type="url_match", evaluator_fn="match_final_url"),
        memory_contract=MemoryContract(
            encodes=browsed, retrieves_from_prior=[],
            must_carry_into_next=[relay_row["url_hash"]]),
        metadata=TaskMetadata(generated_by="llm", generator_seed=seed,
                              generator_version=GENERATOR_VERSION),
    )


def _build_browse_recall(session_idx: int, browse_rows: list[dict],
                         relay_row: dict, recall_target: dict,
                         seed: int) -> TaskInstance:
    """Session 1..4 — browse 20 cross-category products (relay P_N planted),
    then recall the EXACT prior-session product `recall_target` (P_{N-1}) by
    its incidental cue. Scored by url_match against recall_target."""
    recall_cat = recall_target["cat"]
    recall_noun = _category_noun(recall_cat)
    cue = _cue_phrase(recall_target)
    turns = [
        Turn(turn_index=0, role="system", mode="encoding", content="session_start"),
        Turn(turn_index=1, role="user", mode="encoding",
             content="Now I'd like to keep browsing — show me some more pieces."),
    ]
    for i, row in enumerate(browse_rows):
        turns.append(Turn(
            turn_index=2 + i, role="agent", mode="encoding",
            expected_actions=[AgentAction(action_type="navigate", target_url=_url(row))],
            expected_url=_url(row),
        ))
    recall_user_idx = 2 + len(browse_rows)
    recall_agent_idx = recall_user_idx + 1
    turns.append(Turn(
        turn_index=recall_user_idx, role="user", mode="recall",
        content=(
            f"Actually, before I forget — take me back to that one {recall_noun} "
            f"I looked at earlier, the exact one that had a {cue} on it. "
            f"That small detail is how I'll recognise it; I want that specific "
            f"{recall_noun}, not a similar one."
        ),
    ))
    turns.append(Turn(
        turn_index=recall_agent_idx, role="agent", mode="recall",
        expected_actions=[AgentAction(action_type="navigate", target_url=_url(recall_target))],
        expected_url=_url(recall_target),
        recall_turn_metadata=RecallTurnMetadata(
            anchor_invisibility_targets=["page_grid", "recently_viewed"],
            expected_memory_usage="positive"),
    ))
    browsed = [r["url_hash"] for r in browse_rows]
    task_id = f"ic_v2_relay{session_idx}_s{seed:04d}"
    return TaskInstance(
        task_id=task_id, sub_task="IC_xs", grain_tier=1,
        category_ids=_categories_of(browse_rows), variants_used=browsed, turns=turns,
        ncp_metadata=NcpMetadata(
            anchor_variant_ids=[relay_row["url_hash"]],
            recall_turn_indices=[recall_agent_idx],
            memory_gated_branch_turn=recall_agent_idx,
            cross_turn_predicates=[
                f"recall-target: {recall_target['url_hash']} ({recall_cat}, "
                f"cue={recall_target['cue_id']})",
            ],
        ),
        ground_truth=GroundTruth(
            final_action="navigate", target_url=_url(recall_target),
            target_variant_id=recall_target["url_hash"],
            accepted_alternatives=[]),       # EXACT match — no style-loose set
        success_criteria=SuccessCriteria(type="url_match", evaluator_fn="match_final_url"),
        memory_contract=MemoryContract(
            encodes=browsed,
            retrieves_from_prior=[recall_target["url_hash"]],
            must_carry_into_next=[relay_row["url_hash"]]),
        metadata=TaskMetadata(generated_by="llm", generator_seed=seed,
                              generator_version=GENERATOR_VERSION),
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_coord5_ic_v2(
    seed: int,
    val_pool_root: Path | None = None,
    multi_pool_root: Path | None = None,
) -> Tuple[list[TaskInstance], MultiSessionTask]:
    """Build one coord_5 IC v2 relay task. Returns (5 sub-task instances, parent)."""
    rng = random.Random(f"ic_v2.coord5.relay.v2.{seed}")
    registry = json.loads(CUE_REGISTRY_PATH.read_text())["rows"]

    # Pick N_SESSIONS*20 distinct products from the whole catalog and split
    # into 5 browse sets of 20. A uniform sample over 1000 products naturally
    # spans ~all 10 categories, so every session is cross-category. Cues are
    # globally unique => within-task uniqueness is automatic.
    need = N_SESSIONS * BROWSE_PER_SESSION
    pool = rng.sample(registry, need)
    browse_sets: list[list[dict]] = [
        pool[k * BROWSE_PER_SESSION:(k + 1) * BROWSE_PER_SESSION]
        for k in range(N_SESSIONS)
    ]
    # Per session, one of the 20 is the relay product P_k (its cue is recalled
    # by session k+1).
    relays: list[dict] = [
        browse_sets[k][rng.randrange(BROWSE_PER_SESSION)]
        for k in range(N_SESSIONS)
    ]

    sub_tasks: list[TaskInstance] = []

    # Session 0 — Setup (browse 20 cross-category)
    sub_tasks.append(_build_setup_browse(
        browse_rows=browse_sets[0], relay_row=relays[0], seed=seed))

    # Sessions 1..4 — browse 20, recall relays[k-1] (depth-1)
    for k in range(1, N_SESSIONS):
        sub_tasks.append(_build_browse_recall(
            session_idx=k, browse_rows=browse_sets[k],
            relay_row=relays[k], recall_target=relays[k - 1], seed=seed))

    # Persist sub-tasks
    val_root = val_pool_root or V2_VAL_POOL
    for inst in sub_tasks:
        dst = val_root / inst.sub_task / f"{inst.task_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(inst.model_dump_json(indent=2))

    # Compose MultiSessionTask
    refs = []
    for k in range(N_SESSIONS):
        retrieves = [] if k == 0 else [relays[k - 1]["url_hash"]]
        carry = [relays[k]["url_hash"]]
        refs.append(SessionRef(
            task_id=sub_tasks[k].task_id,
            sub_task=("Setup" if k == 0 else "IC_xs"),
            order_index=k,
            retrieves_from_prior_override=retrieves,
            must_carry_override=carry,
            cross_session_user_preamble="",
        ))
    ms_id = f"coord5_ic_v2_relay_s{seed:04d}"
    multi = MultiSessionTask(
        task_id=ms_id, variant="coord_5_ic_v2_relay",
        sessions=refs, cumulative_turn_budget=200,
        cumulative_gt=[relays[k]["url_hash"] for k in range(4)],   # P0..P3
        homogeneous=False,
        metadata={
            "browse_per_session": BROWSE_PER_SESSION,
            "session_categories": [_categories_of(bs) for bs in browse_sets],
            "browse_urls": [[_url(r) for r in bs] for bs in browse_sets],
            "relay_urls": [_url(relays[k]) for k in range(N_SESSIONS)],
            "recall_cues": [
                {"object": relays[k]["cue_object"], "color": relays[k]["cue_color"],
                 "cat": relays[k]["cat"], "url_hash": relays[k]["url_hash"]}
                for k in range(N_SESSIONS)
            ],
            "tier": 1,
            "generator": GENERATOR_VERSION,
        },
    )
    multi_root = multi_pool_root or V2_MULTI_POOL
    save_multi_session(multi, pool=multi_root)
    return sub_tasks, multi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1, help="number of tasks")
    p.add_argument("--seed", type=int, default=0, help="starting seed")
    args = p.parse_args()
    ok = 0
    for i in range(args.n):
        try:
            _, multi = generate_coord5_ic_v2(seed=args.seed + i)
            ok += 1
            cues = multi.metadata["recall_cues"]
            print(f"[ok] {multi.task_id}  "
                  f"relay={'->'.join(c['cat'] for c in cues[:4])}")
        except Exception as e:
            print(f"[skip seed={args.seed + i}] {type(e).__name__}: {e}")
    print(f"generated {ok}/{args.n}")


if __name__ == "__main__":
    _cli()
