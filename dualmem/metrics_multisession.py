"""Multi-session metrics aggregator.

Consumes a list of `MultiSessionResult` (one per task × baseline) and
produces the four headline metric groups defined in
`doc/multisession_design.md §G`:

    1. Task Success Rate (SR)         — fraction fully solved
    2. Progress Score (PS)            — mean per-session correctness
    3. Success Rate at Depth (SR@k)   — paper main figure curve
    4. Information Retention metrics  — IRR@5, MRR@5

Each metric can be stratified by:
    - baseline (memory system name)
    - variant  (short_3 / long_4 / homo_pd / …)
    - sub_task (NC / SA / IC / VL / PD)
    - grain_tier (1 / 2 / 3) — joined from the underlying sub-task JSON

Outputs are CSV-shaped for easy ingestion into the analysis notebooks
(`dualmem/analysis/figures.py` already groups by `system` + `variant`
columns).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from tasks.schema.multisession import MultiSessionResult, SessionResult


# ---------------------------------------------------------------------------
# Row-level dump (one row per task × baseline)
# ---------------------------------------------------------------------------

PER_TASK_COLS = [
    "system", "task_id", "variant", "n_sessions",
    "task_correct", "cumulative_correct", "progress_score",
    "mean_irr_at_5", "mean_mrr_at_5", "mean_set_hit",
    "sr_at_depth_csv",     # "Pass,True,False,..." — per-session; Setup="Pass"
    "elapsed_ms",
]
PER_SESSION_COLS = [
    "system", "task_id", "variant", "session_index", "sub_task",
    "session_correct", "irr_at_5", "mrr_at_5", "set_hit",
    "retrieval_top5_csv",
]


def write_per_task_csv(
    results: Iterable[Tuple[str, MultiSessionResult]],
    path: Path,
) -> int:
    """One row per (system, task). Writes CSV; returns number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_TASK_COLS)
        w.writeheader()
        for system, r in results:
            irrs = [s.irr_at_5 for s in r.sessions if s.irr_at_5 is not None]
            mrrs = [s.mrr_at_5 for s in r.sessions if s.mrr_at_5 is not None]
            sets = [s.retrieval_set_hit for s in r.sessions if s.retrieval_set_hit is not None]
            w.writerow({
                "system": system,
                "task_id": r.task_id,
                "variant": r.variant,
                "n_sessions": len(r.sessions),
                "task_correct": int(r.correct),
                "cumulative_correct": int(r.cumulative_correct),
                "progress_score": round(r.progress_score, 4),
                "mean_irr_at_5": round(mean(irrs), 4) if irrs else -1.0,
                "mean_mrr_at_5": round(mean(mrrs), 4) if mrrs else -1.0,
                "mean_set_hit": round(mean(1.0 if x else 0.0 for x in sets), 4) if sets else -1.0,
                "sr_at_depth_csv": ",".join(
                    "Pass" if s.sub_task == "Setup" else str(s.correct)
                    for s in r.sessions),
                "elapsed_ms": r.elapsed_ms,
            })
            n += 1
    return n


def write_per_session_csv(
    results: Iterable[Tuple[str, MultiSessionResult]],
    path: Path,
) -> int:
    """One row per (system, task, session). Used for SR@k decay aggregation
    and per-sub-task IRR/MRR breakdowns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SESSION_COLS)
        w.writeheader()
        for system, r in results:
            for s in r.sessions:
                w.writerow({
                    "system": system,
                    "task_id": r.task_id,
                    "variant": r.variant,
                    "session_index": s.order_index,
                    "sub_task": s.sub_task,
                    "session_correct": int(s.correct),
                    "irr_at_5": -1.0 if s.irr_at_5 is None else round(s.irr_at_5, 4),
                    "mrr_at_5": -1.0 if s.mrr_at_5 is None else round(s.mrr_at_5, 4),
                    "set_hit": -1 if s.retrieval_set_hit is None else int(s.retrieval_set_hit),
                    "retrieval_top5_csv": ",".join(s.retrieval_top5 or []),
                })
                n += 1
    return n


# ---------------------------------------------------------------------------
# Aggregated tables (paper-ready)
# ---------------------------------------------------------------------------

@dataclass
class BaselineSummary:
    system: str
    n_tasks: int
    sr: float                    # task success rate
    ps: float                    # mean progress score
    sr_at_depth: List[float]     # SR@k for k=1..max_depth; -1 where no task has that depth
    mean_irr_at_5: float         # over all sessions with retrieval contract
    mean_mrr_at_5: float
    mean_set_hit_at_5: float

    def asdict(self) -> dict:
        d = {
            "system": self.system,
            "n_tasks": self.n_tasks,
            "sr": round(self.sr, 4),
            "ps": round(self.ps, 4),
            "mean_irr_at_5": round(self.mean_irr_at_5, 4),
            "mean_mrr_at_5": round(self.mean_mrr_at_5, 4),
            "mean_set_hit_at_5": round(self.mean_set_hit_at_5, 4),
        }
        for k, v in enumerate(self.sr_at_depth, start=1):
            d[f"sr_at_{k}"] = round(v, 4) if v >= 0 else -1.0
        return d


def aggregate(
    results: List[Tuple[str, MultiSessionResult]],
    max_depth: int = 4,
) -> List[BaselineSummary]:
    """Group results by `system` and compute the four metric groups.

    SR@k aggregation rule: a task with N sessions contributes to SR@k only
    for k ≤ N (k is 1-indexed). The denominator at depth k is the count of
    tasks with ≥ k sessions.
    """
    by_sys: Dict[str, List[MultiSessionResult]] = defaultdict(list)
    for system, r in results:
        by_sys[system].append(r)

    # Only NON-Setup sessions are scored SUBTASKS. Browse/Setup sessions are
    # an encoding phase (trivially "correct") — counting them would inflate
    # PS and shift the SR@depth axis. The k-th recall subtask is depth k.
    def _subtasks(r: MultiSessionResult) -> list:
        return [s for s in r.sessions if s.sub_task != "Setup"]

    out: List[BaselineSummary] = []
    for system, rs in sorted(by_sys.items()):
        n_tasks = len(rs)
        # SR = every recall subtask correct;  PS = fraction of recalls correct.
        sr = mean(1.0 if (st := _subtasks(r)) and all(s.correct for s in st)
                  else 0.0 for r in rs) if rs else 0.0
        ps = mean((mean(1.0 if s.correct else 0.0 for s in st) if (st := _subtasks(r))
                   else 0.0) for r in rs) if rs else 0.0

        # SR@k decay — cumulative interpretation:
        # fraction of tasks where recall subtasks 1..k are ALL correct.
        # Recall subtask k has horizon depth k. Monotonically non-increasing.
        sr_at_k: List[float] = []
        for k in range(1, max_depth + 1):
            num = 0
            den = 0
            for r in rs:
                st = _subtasks(r)
                if len(st) >= k:
                    den += 1
                    if all(st[j].correct for j in range(k)):
                        num += 1
            sr_at_k.append(num / den if den else -1.0)

        # IRR/MRR over all sessions with retrieval contract
        irrs: List[float] = []
        mrrs: List[float] = []
        hits: List[float] = []
        for r in rs:
            for s in r.sessions:
                if s.irr_at_5 is not None: irrs.append(s.irr_at_5)
                if s.mrr_at_5 is not None: mrrs.append(s.mrr_at_5)
                if s.retrieval_set_hit is not None:
                    hits.append(1.0 if s.retrieval_set_hit else 0.0)

        out.append(BaselineSummary(
            system=system,
            n_tasks=n_tasks,
            sr=sr,
            ps=ps,
            sr_at_depth=sr_at_k,
            mean_irr_at_5=mean(irrs) if irrs else -1.0,
            mean_mrr_at_5=mean(mrrs) if mrrs else -1.0,
            mean_set_hit_at_5=mean(hits) if hits else -1.0,
        ))
    return out


def write_summary_csv(summary: List[BaselineSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        return
    rows = [s.asdict() for s in summary]
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def format_summary_table(summary: List[BaselineSummary], max_depth: int = 4) -> str:
    """Console-friendly summary table for quick inspection."""
    lines = []
    hdr = f"{'system':12s} {'n':>4s} {'SR':>6s} {'PS':>6s} {'IRR@5':>7s} {'MRR@5':>7s} {'hit@5':>7s}"
    for k in range(1, max_depth + 1):
        hdr += f" {'SR@'+str(k):>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in summary:
        line = (f"{s.system:12s} {s.n_tasks:>4d} {s.sr:>6.2%} {s.ps:>6.2%} "
                f"{s.mean_irr_at_5:>7.2%} {s.mean_mrr_at_5:>7.2%} "
                f"{s.mean_set_hit_at_5:>7.2%}")
        for v in s.sr_at_depth:
            line += f" {v:>7.2%}" if v >= 0 else f" {'-':>7s}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variant / sub_task stratifications (optional drill-down)
# ---------------------------------------------------------------------------

def aggregate_by_variant(
    results: List[Tuple[str, MultiSessionResult]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Returns {(system, variant): {sr, ps, mean_irr_at_5, mean_mrr_at_5, n}}."""
    by_key: Dict[Tuple[str, str], List[MultiSessionResult]] = defaultdict(list)
    for system, r in results:
        by_key[(system, r.variant)].append(r)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (system, variant), rs in by_key.items():
        irrs = [s.irr_at_5 for r in rs for s in r.sessions if s.irr_at_5 is not None]
        mrrs = [s.mrr_at_5 for r in rs for s in r.sessions if s.mrr_at_5 is not None]
        out[(system, variant)] = {
            "n": len(rs),
            "sr": mean(1.0 if r.correct else 0.0 for r in rs),
            "ps": mean(r.progress_score for r in rs),
            "mean_irr_at_5": mean(irrs) if irrs else -1.0,
            "mean_mrr_at_5": mean(mrrs) if mrrs else -1.0,
        }
    return out
