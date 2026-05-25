"""Unified runner: dispatches TaskCells to the right ceiling × baseline runner.

Public API:
    run_cells(cells, manifest, vlm, systems, distractor_pool=None) -> List[CellResult]
    write_results(results, out_path)
    load_results(path) -> List[dict]
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dualmem.types import TaskCell, CellResult
from dualmem.inventory import InventoryManifest, make_trial
from dualmem.systems import make_system
from dualmem.ceiling import run_perception, run_oracle_retrieval, run_full_pipeline


def _sample_distractor_memories(cell: TaskCell, manifest: InventoryManifest,
                                 n: int = 5, rng: random.Random = None) -> List:
    """Pick K distractor trials from OTHER categories at the same grain tier."""
    rng = rng or random.Random(cell.seed)
    other_cats = [c for c in manifest.products if c != cell.category]
    rng.shuffle(other_cats)
    out = []
    for cat in other_cats[:n]:
        variants = list(manifest.products[cat].keys())
        v = rng.choice(variants)
        dummy_cell = TaskCell(
            category=cat, variant=v,
            grain_tier=cell.grain_tier, leakage_level=cell.leakage_level,
            mechanism=cell.mechanism, ceiling="full_pipeline",
            seed=cell.seed,
        )
        out.append(make_trial(dummy_cell, manifest))
    return out


def run_cells(
    cells: Iterable[TaskCell],
    manifest: InventoryManifest,
    vlm,
    systems: List[str],
    distractor_count: int = 5,
    on_result: Optional[callable] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 20,
    system_kwargs: Optional[Dict] = None,
) -> List[CellResult]:
    """Run a batch of cells. Returns the flat list of CellResults.

    For perception ceiling: a single VLM call per cell (system_name=Perception).
    For oracle_retrieval / full_pipeline: one VLM call per (cell, system) pair.
    """
    results: List[CellResult] = []
    t_start = time.time()
    cells = list(cells)
    n_total = 0
    for c in cells:
        if c.ceiling == "perception":
            n_total += 1
        else:
            n_total += len(systems)

    # Build systems ONCE per run so heavy encoder loads (CLIP, SBERT) happen
    # once and the bank.reset() at each trial start clears the entries.
    system_kwargs = system_kwargs or {}
    system_pool = {name: make_system(name, **system_kwargs) for name in systems}

    done = 0

    def _maybe_ckpt():
        if checkpoint_path and done > 0 and done % checkpoint_every == 0:
            write_results(results, checkpoint_path)

    for cell in cells:
        trial = make_trial(cell, manifest)
        if cell.ceiling == "perception":
            r = run_perception(trial, vlm)
            results.append(r)
            done += 1
            if verbose and done % max(1, n_total // 50) == 0:
                _log_progress(done, n_total, t_start)
            if on_result: on_result(r)
            _maybe_ckpt()
        elif cell.ceiling == "oracle_retrieval":
            for sys_name in systems:
                system = system_pool[sys_name]
                system.reset()
                # Encode anchor so oracle_inject can pull a cached caption / embedding.
                system.encode(trial)
                r = run_oracle_retrieval(trial, system, vlm)
                results.append(r)
                done += 1
                if verbose and done % max(1, n_total // 20) == 0:
                    _log_progress(done, n_total, t_start)
                if on_result: on_result(r)
        elif cell.ceiling == "full_pipeline":
            distractors = _sample_distractor_memories(cell, manifest, n=distractor_count)
            for sys_name in systems:
                system = system_pool[sys_name]
                # `system.reset()` is called inside run_full_pipeline.
                r = run_full_pipeline(trial, system, vlm, distractor_memories=distractors)
                results.append(r)
                done += 1
                if verbose and done % max(1, n_total // 20) == 0:
                    _log_progress(done, n_total, t_start)
                if on_result: on_result(r)
        else:
            raise ValueError(f"Unknown ceiling: {cell.ceiling}")
    if verbose:
        _log_progress(done, n_total, t_start)
    if checkpoint_path:
        write_results(results, checkpoint_path)
    return results


def _log_progress(done, total, t_start):
    elapsed = time.time() - t_start
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    print(f"  [{done}/{total}]  elapsed={elapsed:6.1f}s  rate={rate:5.2f}/s  eta={eta:6.1f}s", flush=True)


def write_results(results: List[CellResult], out_path: str) -> None:
    """Write CellResults to CSV + JSONL. CSV is for pandas; JSONL keeps raw_response."""
    out_csv = Path(out_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_csv.with_suffix(".jsonl")

    if not results:
        return

    rows = [r.asdict() for r in results]
    # Union of keys across rows (different ceilings carry different extras).
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    with open(out_jsonl, "w") as f:
        for r in results:
            d = r.asdict()
            d["raw_response"] = r.raw_response
            f.write(json.dumps(d) + "\n")


def load_results(path: str) -> List[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    with open(p) as f:
        return list(csv.DictReader(f))
