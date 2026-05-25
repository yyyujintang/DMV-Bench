"""Layer-gap metrics for VisMem-Diag.

Three-layer decomposition (Proposal_A, Axis 3):
    perception_acc    : VLM's pure visual discrimination
    oracle_acc(sys)   : ceiling for memory system `sys` (skips retrieval)
    full_acc(sys)     : end-to-end accuracy for `sys`

Layer gaps:
    perception_gap(sys) = perception_acc - oracle_acc(sys)   # injection-format loss
    retrieval_gap(sys)  = oracle_acc(sys) - full_acc(sys)    # retrieval loss
    total_gap(sys)      = perception_acc - full_acc(sys)     # composite

We aggregate by (grain_tier × leakage_level × mechanism) so the 2D heatmap
can show where each system breaks. CIs come from non-parametric bootstrap.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def accuracy(results: Iterable[dict], filter_fn=None) -> Tuple[float, int]:
    rs = [r for r in results if filter_fn is None or filter_fn(r)]
    if not rs:
        return float("nan"), 0
    correct = sum(1 for r in rs if int(r.get("correct", 0)) == 1)
    return correct / len(rs), len(rs)


def bootstrap_ci(results: Iterable[dict], filter_fn=None, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float, int]:
    rs = [r for r in results if filter_fn is None or filter_fn(r)]
    if not rs:
        return float("nan"), float("nan"), float("nan"), 0
    values = np.array([int(r.get("correct", 0)) for r in rs])
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi), len(values)


def by_ceiling_and_system(results: Iterable[dict]) -> Dict[Tuple[str, str], List[dict]]:
    g = defaultdict(list)
    for r in results:
        g[(r["ceiling"], r["system"])].append(r)
    return dict(g)


def layer_gap_table(results: Iterable[dict], systems: List[str]) -> List[dict]:
    """Compute per-system perception/oracle/full + gaps. Returns list of dicts."""
    rs = list(results)
    p_acc, p_n = accuracy(rs, lambda r: r["ceiling"] == "perception")
    out = []
    for sys in systems:
        o_acc, o_n = accuracy(rs, lambda r: r["ceiling"] == "oracle_retrieval" and r["system"] == sys)
        f_acc, f_n = accuracy(rs, lambda r: r["ceiling"] == "full_pipeline" and r["system"] == sys)
        out.append({
            "system": sys,
            "perception_acc": p_acc, "perception_n": p_n,
            "oracle_acc": o_acc, "oracle_n": o_n,
            "full_acc": f_acc, "full_n": f_n,
            "injection_gap": (p_acc - o_acc) if not math.isnan(o_acc) else float("nan"),
            "retrieval_gap": (o_acc - f_acc) if (not math.isnan(o_acc) and not math.isnan(f_acc)) else float("nan"),
            "total_gap": (p_acc - f_acc) if not math.isnan(f_acc) else float("nan"),
        })
    return out


def stratified_accuracy(results: Iterable[dict], stratify_keys: List[str]) -> Dict[tuple, dict]:
    """Compute accuracy grouped by stratification keys.

    Returns mapping (key1_value, key2_value, ...) → {ceiling: {system: (acc, n)}}.
    """
    out = defaultdict(lambda: defaultdict(dict))
    groups = defaultdict(list)
    for r in results:
        key = tuple(r.get(k) for k in stratify_keys)
        groups[key].append(r)
    for key, rs in groups.items():
        ceil_sys = defaultdict(list)
        for r in rs:
            ceil_sys[(r["ceiling"], r["system"])].append(r)
        for (ceil, sys), batch in ceil_sys.items():
            acc, n = accuracy(batch)
            out[key][ceil][sys] = (acc, n)
    return dict(out)
