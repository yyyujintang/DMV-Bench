"""Family-2 metrics — aggregate a spine+probes run (advisor's final design).

Consumes `{system_name: [F2TaskResult, ...]}` (one F2TaskResult per task spine)
and reports:
  - R@reach — recall accuracy at each reach (sessions back the cue was injected).
    This SR-vs-reach retention curve is the headline long-horizon result.
  - task_SR — fraction of task spines on which EVERY recall probe is correct.
  - inj_ok — fraction of expected cues the online edit actually placed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List


def aggregate_f2(tasks: List, results_by_system: Dict[str, List],
                 out_dir: Path | None = None) -> str:
    """Build the Family-2 summary table. Returns it as text; writes CSV if out_dir."""
    max_reach = max((p.reach for t in tasks for p in t.probes), default=0)
    expected_cues = sum(t.n_sessions for t in tasks)
    rows = []
    # A recall (cue -> url_match) IS the task. SR = success over all recall
    # tasks; R@reach = SR by reach. (chain_SR removed — uninformative at scale.)
    hdr = (f"{'system':14s}{'SR':>8s}{'inj_ok':>8s}"
           f"{'n_recall':>10s}"
           + "".join(f"{'R@r' + str(r):>8s}" for r in range(1, max_reach + 1)))
    lines = [hdr, "-" * len(hdr)]

    for system in sorted(results_by_system):
        task_results = results_by_system[system]
        all_probes = [p for tr in task_results for p in tr.probes]
        placed = sum(len(tr.injections) for tr in task_results)
        inj_ok = placed / expected_cues if expected_cues else 1.0
        overall = (mean(1.0 if p.correct else 0.0 for p in all_probes)
                   if all_probes else 0.0)
        per_reach = []
        for reach in range(1, max_reach + 1):
            vals = [1.0 if p.correct else 0.0
                    for p in all_probes if p.reach == reach]
            per_reach.append(mean(vals) if vals else -1.0)
        lines.append(f"{system:14s}{overall * 100:>7.1f}%"
                     f"{inj_ok * 100:>7.1f}%{len(all_probes):>10d}"
                     + "".join(f"{v * 100:>7.1f}%" for v in per_reach))
        rows.append({"system": system, "SR": round(overall, 4),
                     "injection_placed": round(inj_ok, 4),
                     "n_recall_tasks": len(all_probes),
                     **{f"recall_at_reach_{r}": round(per_reach[r - 1], 4)
                        for r in range(1, max_reach + 1)}})

    table = "\n".join(lines)
    if out_dir is not None and rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "f2_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (out_dir / "f2_summary.txt").write_text(table + "\n")
        with (out_dir / "f2_per_probe.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "task_id", "probe_id", "at_session",
                        "target_session", "reach", "correct",
                        "target_url", "final_url", "vlm_calls"])
            for system, task_results in results_by_system.items():
                for tr in task_results:
                    for p in tr.probes:
                        w.writerow([system, tr.task_id, p.probe_id, p.at_session,
                                    p.target_session, p.reach, int(p.correct),
                                    p.target_url or "", p.final_url or "",
                                    p.vlm_calls])
    return table
