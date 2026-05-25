"""Family-2 generator — Per-Product IC (design v6).

Emits ONE `RolloutTreeTask` (a linear spine, J-INDEPENDENT):

  - a linear chain of D sessions; session j is one WebArena-style comparison
    shopping task (~22-28 steps, one category);
  - NO pre-chosen cue per session — every product the agent autonomously
    views already carries a unique visual cue baked into the storefront image
    (the website serves `images/with_cue/<cat>/<style>/<idx>.png`, and
    `data/vismem_diag_v2/cue_registry.json` records the (object, color) per
    url_hash);
  - recall probes are FILLED AT RUN TIME from the recorded encoding
    trajectory (one probe per (at_session j, target_session j-r, viewed
    product u)), so |IC|/chain = k·N and |Q|/chain ≈ k·N(N-1)/2.

See doc/f2_task_design_v6.png.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schema.rollout_tree_task import (
    RecallProbe, RolloutTreeTask, SessionSpec, save_rollout_tree,
)

GENERATOR_VERSION = "f2.per_product_ic.v6"

REPO_ROOT = Path(__file__).resolve().parents[2]
F2_POOL = REPO_ROOT / "tasks" / "pool_v2" / "f2_trees"

CATEGORIES = ["chair", "sofa", "lamp", "cushion", "vase",
              "rug", "table", "bookshelf", "plant_pot", "wall_art"]


def generate_f2_chain(
    seed: int = 0,
    n_sessions: int = 5,
    max_steps: int = 50,
    pool_root: Path | None = None,
) -> RolloutTreeTask:
    """Build one Family-2 (v6) task — a linear D-session spine.

    Session specs are J-INDEPENDENT: session j depends only on (seed, j),
    never on n_sessions. So a J=5 task and a J=10 task with the same seed
    share sessions 0-4 byte-for-byte. The `probes` list is left EMPTY here —
    it is filled at run time once the agent has actually browsed (since which
    products carry the recall targets is autonomous)."""
    # category order depends ONLY on seed (J-independent)
    seed_rng = random.Random(f"f2.seed.{seed}")
    cat_order = seed_rng.sample(CATEGORIES, len(CATEGORIES))

    sessions: list[SessionSpec] = []
    for j in range(n_sessions):
        # per-session RNG — J-independent
        srng = random.Random(f"f2.session.{seed}.{j}")
        # ~22-28 steps comparison-shopping → ~12 distinct products / session
        n_steps = srng.randint(22, 28)
        sessions.append(SessionSpec(
            session_idx=j,
            shopping_list=[cat_order[j % len(CATEGORIES)]],
            n_steps=n_steps,
        ))

    task = RolloutTreeTask(
        task_id=f"f2_chain_d{n_sessions}_s{seed:04d}",
        n_sessions=n_sessions, max_steps_per_session=max_steps,
        sessions=sessions, probes=[],   # filled at run time from trajectories
        metadata={
            "generator": GENERATOR_VERSION, "seed": seed,
            "design": "v6_per_product_ic",
        },
    )
    save_rollout_tree(task, pool_root or F2_POOL)
    return task


def build_probes_from_trajectories(
    task: RolloutTreeTask,
    trajectories_by_session: dict[int, list[tuple[str, str]]],
    cue_lookup: dict[str, dict],
    mc_probes: int = 0,
    mc_seed: str = "f2.mc.default",
) -> list[RecallProbe]:
    """Fill `task.probes` at run time from the recorded encoding trajectories.

    Two modes:

    1. **Exhaustive** (`mc_probes <= 0`, default):
       For each session j in 1..N-1, for each reach r in 1..j, for each
       distinct product u viewed in session j-r → emit one RecallProbe.
       Probe count = k · N(N-1)/2 per chain, quadratic in N.

    2. **Monte Carlo** (`mc_probes > 0`):
       For each reach r in 1..N-1, gather all (at_session j, target_session
       j-r, viewed_product u) tuples across the chain, and randomly sample
       `mc_probes` of them (or all if fewer exist). Sampling is seeded
       deterministically by `(mc_seed, task_id, reach)` so any rerun with
       the same args produces byte-identical probes — `f2_per_probe.csv`
       is reproducible. Total probes per chain ≈ `mc_probes × (N-1)`,
       LINEAR in N — lets us push to J=50, J=100 cheaply for sparse
       long-horizon coverage.

    The cue (object, color, id) per probe is read from `cue_lookup[u]`
    (= `cue_registry.json` rows). Schema of emitted `RecallProbe` is
    identical between modes; downstream `f2_per_probe.csv` consumes both.
    """
    import random

    probes: list[RecallProbe] = []
    # de-dup visited per session (preserve first-seen order)
    visited_per_session: dict[int, list[str]] = {}
    for j, visits in trajectories_by_session.items():
        seen: list[str] = []
        seen_set: set[str] = set()
        for uh, _img in visits:
            if uh in seen_set:
                continue
            if uh not in cue_lookup:
                continue   # product not in registry; skip
            seen.append(uh)
            seen_set.add(uh)
        visited_per_session[j] = seen

    if mc_probes <= 0:
        # Exhaustive (unchanged behaviour) — quadratic |Q| = k·N(N-1)/2.
        for j in range(1, task.n_sessions):
            for r in range(1, j + 1):
                tgt = j - r
                for p_i, uh in enumerate(visited_per_session.get(tgt, [])):
                    c = cue_lookup[uh]
                    probes.append(RecallProbe(
                        probe_id=f"s{j}_r{r}_p{p_i:02d}",
                        at_session=j, target_session=tgt,
                        target_url_hash=uh,
                        target_cue_id=c["cue_id"],
                        target_cue_object=c["cue_object"],
                        target_cue_color=c["cue_color"],
                        reach=r))
        return probes

    # Monte Carlo — sample `mc_probes` uniformly per reach.
    for r in range(1, task.n_sessions):
        # collect every eligible (at_session j, target_session tgt, viewed_product u)
        # triple for this reach, across the whole chain.
        candidates: list[tuple[int, int, str]] = []
        for j in range(r, task.n_sessions):
            tgt = j - r
            for uh in visited_per_session.get(tgt, []):
                candidates.append((j, tgt, uh))
        if not candidates:
            continue
        rng = random.Random(f"{mc_seed}.{task.task_id}.reach{r}")
        n_sample = min(mc_probes, len(candidates))
        sampled = rng.sample(candidates, n_sample)
        for p_i, (j, tgt, uh) in enumerate(sampled):
            c = cue_lookup[uh]
            probes.append(RecallProbe(
                probe_id=f"s{j}_r{r}_p{p_i:03d}_mc",
                at_session=j, target_session=tgt,
                target_url_hash=uh,
                target_cue_id=c["cue_id"],
                target_cue_object=c["cue_object"],
                target_cue_color=c["cue_color"],
                reach=r))
    return probes


# Back-compat alias for the entry point.
generate_f2_tree = generate_f2_chain


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sessions", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=50)
    args = p.parse_args()
    t = generate_f2_chain(seed=args.seed, n_sessions=args.n_sessions,
                          max_steps=args.max_steps)
    print(f"[ok] {t.task_id}  sessions={t.n_sessions}  "
          f"(probes filled at run time)")


if __name__ == "__main__":
    _cli()
