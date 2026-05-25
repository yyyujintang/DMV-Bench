"""Figure generation for VisMem-Diag.

Three core figures (Proposal_A deliverables):
  1. layer_gap_figure: per-system perception/oracle/full bar chart with gaps.
  2. cube_figure: (perception × retrieval × injection) cube — each system as a point.
  3. granularity_leakage_heatmap: 2D map of accuracy by grain_tier × leakage_level.

All figures use matplotlib only, no seaborn dep, headless backend.
"""

from __future__ import annotations

import csv
import math
import os
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(path: str) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _by(rows, key, val, val_is_int=False):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(int(r[val]) if val_is_int else r[val])
    return out


def _acc(rs, filter_fn):
    rs2 = [r for r in rs if filter_fn(r)]
    if not rs2: return float("nan"), 0
    return sum(int(r["correct"]) for r in rs2) / len(rs2), len(rs2)


def _wilson(p, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - rad, centre + rad


# ---------------------------------------------------------------------------
# Figure 1: per-system layer gap bar chart
# ---------------------------------------------------------------------------

def layer_gap_figure(results_path: str, out_path: str, systems: Optional[List[str]] = None,
                     title: str = "VisMem-Diag — Three-Layer Decomposition"):
    rows = _load_results(results_path)
    if systems is None:
        systems = sorted({r["system"] for r in rows if r["ceiling"] != "perception"})

    perception_acc, perception_n = _acc(rows, lambda r: r["ceiling"] == "perception")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(systems))
    width = 0.35

    oracle_accs, full_accs, oracle_cis, full_cis = [], [], [], []
    for sys in systems:
        a, n = _acc(rows, lambda r: r["ceiling"] == "oracle_retrieval" and r["system"] == sys)
        oracle_accs.append(a); oracle_cis.append(_wilson(a, n))
        a, n = _acc(rows, lambda r: r["ceiling"] == "full_pipeline" and r["system"] == sys)
        full_accs.append(a); full_cis.append(_wilson(a, n))

    o_lo = [a - ci[0] for a, ci in zip(oracle_accs, oracle_cis)]
    o_hi = [ci[1] - a for a, ci in zip(oracle_accs, oracle_cis)]
    f_lo = [a - ci[0] for a, ci in zip(full_accs, full_cis)]
    f_hi = [ci[1] - a for a, ci in zip(full_accs, full_cis)]

    b1 = ax.bar(x - width/2, oracle_accs, width, yerr=[o_lo, o_hi], capsize=4,
                label="Oracle-retrieval", color="#3a86ff")
    b2 = ax.bar(x + width/2, full_accs, width, yerr=[f_lo, f_hi], capsize=4,
                label="Full pipeline", color="#fb5607")

    ax.axhline(perception_acc, color="#2a9d8f", linestyle="--", linewidth=2,
               label=f"Perception ceiling = {perception_acc:.3f} (n={perception_n})")
    ax.axhline(0.25, color="#888", linestyle=":", linewidth=1, label="Chance (4AFC)")

    ax.set_ylabel("4AFC accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: (perception × retrieval × injection) cube
# ---------------------------------------------------------------------------

def cube_figure(results_path: str, out_path: str, systems: Optional[List[str]] = None,
                title: str = "VisMem-Diag — (Perception × Retrieval × Injection) Cube"):
    """Plot each baseline as a point in (injection_gap, retrieval_gap, full_acc) space.

    To stay 2D-readable we project onto (injection_gap on x, retrieval_gap on y)
    and color points by full_acc. Small values of both gaps = method is good.
    """
    rows = _load_results(results_path)
    if systems is None:
        systems = sorted({r["system"] for r in rows if r["ceiling"] != "perception"})

    p_acc, _ = _acc(rows, lambda r: r["ceiling"] == "perception")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    points_x, points_y, points_z, labels = [], [], [], []
    for sys in systems:
        o_acc, _ = _acc(rows, lambda r: r["ceiling"] == "oracle_retrieval" and r["system"] == sys)
        f_acc, _ = _acc(rows, lambda r: r["ceiling"] == "full_pipeline" and r["system"] == sys)
        inj_gap = p_acc - o_acc
        ret_gap = o_acc - f_acc
        points_x.append(inj_gap); points_y.append(ret_gap); points_z.append(f_acc); labels.append(sys)

    sc = ax.scatter(points_x, points_y, c=points_z, cmap="viridis", s=180,
                    vmin=0.25, vmax=max(1.0, max(points_z) if points_z else 1.0),
                    edgecolor="black", linewidth=0.8, zorder=3)
    for x_, y_, lbl in zip(points_x, points_y, labels):
        ax.annotate(lbl, (x_, y_), xytext=(7, 7), textcoords="offset points",
                    fontsize=10, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, label="Full-pipeline accuracy")

    ax.set_xlabel("Injection gap  (perception − oracle-retrieval)")
    ax.set_ylabel("Retrieval gap  (oracle-retrieval − full-pipeline)")
    ax.set_title(title)
    ax.axhline(0, color="#aaa", linewidth=0.5); ax.axvline(0, color="#aaa", linewidth=0.5)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: 2D heatmap (granularity × leakage)
# ---------------------------------------------------------------------------

def granularity_leakage_heatmap(results_path: str, out_path: str,
                                ceiling: str = "full_pipeline",
                                system: Optional[str] = None):
    rows = _load_results(results_path)
    rows = [r for r in rows if r["ceiling"] == ceiling]
    if system: rows = [r for r in rows if r["system"] == system]
    if not rows:
        print(f"[heatmap] no rows for ceiling={ceiling} system={system}; skipping")
        return None

    tiers = sorted({int(r["grain_tier"]) for r in rows})
    levels = sorted({int(r["leakage_level"]) for r in rows})

    M = np.full((len(tiers), len(levels)), np.nan)
    Ns = np.zeros_like(M, dtype=int)
    for i, t in enumerate(tiers):
        for j, l in enumerate(levels):
            xs = [r for r in rows if int(r["grain_tier"]) == t and int(r["leakage_level"]) == l]
            if xs:
                M[i, j] = sum(int(r["correct"]) for r in xs) / len(xs)
                Ns[i, j] = len(xs)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(M, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_yticks(range(len(tiers))); ax.set_yticklabels([f"tier{t}" for t in tiers])
    ax.set_xlabel("Text leakage level (0=masked → 4=full marketing)")
    ax.set_ylabel("Visual grain tier (1=coarse → 3=fine)")
    sub = f"system={system or 'all'}, ceiling={ceiling}"
    ax.set_title(f"VisMem-Diag — accuracy by granularity × leakage\n{sub}")
    for i in range(len(tiers)):
        for j in range(len(levels)):
            v = M[i, j]; n = Ns[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:.2f}\nn={n}", ha="center", va="center",
                    color="black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary table for write-up
# ---------------------------------------------------------------------------

def leakage_comparison_figure(results_path: str, out_path: str,
                              systems: Optional[List[str]] = None,
                              ceilings: List[str] = ("oracle_retrieval", "full_pipeline")):
    """Grouped bar chart: per-system accuracy at each leakage level, faceted by ceiling.

    Tests Proposal_A's Axis 2 prediction: as leakage rises, text-based memories
    gain (text now carries category signal), but visual memories should be
    unaffected at the injection layer.
    """
    rows = _load_results(results_path)
    if systems is None:
        systems = sorted({r["system"] for r in rows if r["ceiling"] != "perception"})
    levels = sorted({int(r["leakage_level"]) for r in rows if r["ceiling"] != "perception"})

    n_ceil = len(ceilings)
    fig, axes = plt.subplots(1, n_ceil, figsize=(5.5 * n_ceil, 4.2), sharey=True)
    if n_ceil == 1: axes = [axes]

    x = np.arange(len(systems))
    width = 0.8 / max(1, len(levels))
    cmap = plt.cm.viridis

    for ax, ceil in zip(axes, ceilings):
        for i, lvl in enumerate(levels):
            accs, errs_lo, errs_hi = [], [], []
            for sys in systems:
                a, n = _acc(rows, lambda r: r["ceiling"] == ceil and r["system"] == sys and int(r["leakage_level"]) == lvl)
                lo, hi = _wilson(a, n)
                accs.append(a)
                errs_lo.append(max(0, a - lo)); errs_hi.append(max(0, hi - a))
            offset = (i - (len(levels) - 1) / 2) * width
            ax.bar(x + offset, accs, width, yerr=[errs_lo, errs_hi], capsize=3,
                   label=f"L{lvl}", color=cmap(i / max(1, len(levels) - 1)))
        ax.axhline(0.25, color="#888", linestyle=":", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(systems, rotation=10)
        ax.set_ylim(0, 1.05); ax.set_title(f"ceiling = {ceil}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ax is axes[0]: ax.set_ylabel("4AFC accuracy")
        ax.legend(loc="upper right", title="leakage", framealpha=0.9, fontsize=9)

    fig.suptitle("VisMem-Diag — accuracy × leakage level", fontsize=12)
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def summary_table_to_markdown(results_path: str, systems: Optional[List[str]] = None) -> str:
    rows = _load_results(results_path)
    if systems is None:
        systems = sorted({r["system"] for r in rows if r["ceiling"] != "perception"})

    p_acc, p_n = _acc(rows, lambda r: r["ceiling"] == "perception")
    lines = [
        "| Memory system | Perception | Oracle-retrieval | Full pipeline | Inj gap | Retr gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sys in systems:
        o, _ = _acc(rows, lambda r: r["ceiling"] == "oracle_retrieval" and r["system"] == sys)
        f, _ = _acc(rows, lambda r: r["ceiling"] == "full_pipeline" and r["system"] == sys)
        inj = p_acc - o
        ret = o - f
        lines.append(f"| {sys} | {p_acc:.3f} (n={p_n}) | {o:.3f} | {f:.3f} | {inj:+.3f} | {ret:+.3f} |")
    return "\n".join(lines)
