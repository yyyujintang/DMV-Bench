"""Incidental-Cue chain generator.

Emits one `ChainTask`: a linear chain of D sessions, each a short WebArena-style
comparison-shopping task over one category (22-28 steps). Session j depends only
on (seed, j), never on D, so chains of different lengths built from the same
seed share their early sessions byte-for-byte -- this is what lets a J=15 run
reuse every session of the J=5 run, and what makes the comparison across memory
systems paired.

No cue is chosen here. Every product the agent may open already carries a unique
visual cue baked into the storefront image, recorded per url_hash in
`data/vismem_diag_v2/cue_registry.json`. Recall probes are therefore filled in at
run time from the recorded trajectory, one per (probe session, target session,
viewed product).
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schema.chain_task import (
    ChainTask, RecallProbe, SessionSpec, save_chain_task,
)

GENERATOR_VERSION = "ic.chain.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_POOL = REPO_ROOT / "tasks" / "pool_v2" / "f2_trees"

CATEGORIES = ["chair", "sofa", "lamp", "cushion", "vase",
              "rug", "table", "bookshelf", "plant_pot", "wall_art"]


def generate_f2_chain(
    seed: int = 0,
    n_sessions: int = 5,
    pool_root: Path | None = None,
) -> ChainTask:
    """Build one D-session chain.

    Session specs depend ONLY on (seed, j), never on n_sessions, so a J=5 and a
    J=10 chain with the same seed share sessions 0-4 byte-for-byte. `probes` is
    left empty here and filled at run time once the agent has actually browsed."""
    # category order depends ONLY on seed (length-independent)
    seed_rng = random.Random(f"f2.seed.{seed}")
    cat_order = seed_rng.sample(CATEGORIES, len(CATEGORIES))

    sessions: list[SessionSpec] = []
    for j in range(n_sessions):
        # per-session RNG -- length-independent
        srng = random.Random(f"f2.session.{seed}.{j}")
        # 22-28 steps of comparison shopping -> ~12 distinct products / session
        n_steps = srng.randint(22, 28)
        sessions.append(SessionSpec(
            session_idx=j,
            shopping_list=[cat_order[j % len(CATEGORIES)]],
            n_steps=n_steps,
        ))

    task = ChainTask(
        task_id=f"f2_chain_d{n_sessions}_s{seed:04d}",
        n_sessions=n_sessions,
        sessions=sessions, probes=[],   # filled at run time from trajectories
        metadata={"generator": GENERATOR_VERSION, "seed": seed},
    )
    save_chain_task(task, pool_root or CHAIN_POOL)
    return task


def build_probes_from_trajectories(
    task: ChainTask,
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


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sessions", type=int, default=5)
    args = p.parse_args()
    t = generate_f2_chain(seed=args.seed, n_sessions=args.n_sessions)
    print(f"[ok] {t.task_id}  sessions={t.n_sessions}  "
          f"(probes filled at run time)")


if __name__ == "__main__":
    _cli()
