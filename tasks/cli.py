"""
Task pipeline CLI — `python -m tasks.cli {generate,validate,pool-stats}`.

Generate emits TaskInstance JSON to pool/generated/<sub-task>/.
Validate moves each JSON to pool/validated/ or pool/rejected/<id>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .generators._common import VariantCatalogue
from .generators import (
    nc_negative_constraint as nc,
    sa_style_abstraction as sa,
    ic_incidental_cue as ic,
    vl_visual_landmark as vl,
    pd_preference_drift as pd,
    pd_long,
)
from .schema.task_instance import TaskInstance
from .validators.ncp_validator import validate_task


def _pd_long_short(cat, seed, **kw): return pd_long.generate(cat, seed, length="short", **kw)
def _pd_long_medium(cat, seed, **kw): return pd_long.generate(cat, seed, length="medium", **kw)
def _pd_long_long(cat, seed, **kw): return pd_long.generate(cat, seed, length="long", **kw)


GENERATORS = {
    "NC": nc.generate,
    "SA": sa.generate,
    "IC": ic.generate,
    "VL": vl.generate,
    "PD": pd.generate,
    "PD_short": _pd_long_short,
    "PD_medium": _pd_long_medium,
    "PD_long": _pd_long_long,
}

# PD-long variants land in the same on-disk SUB dir as PD (filename prefix
# distinguishes them; composer + listAllTaskIds filter on prefix).
SUB_DIR_FOR_KIND = {
    "NC": "NC", "SA": "SA", "IC": "IC", "VL": "VL", "PD": "PD",
    "PD_short": "PD", "PD_medium": "PD", "PD_long": "PD",
}

POOL = Path(__file__).resolve().parent / "pool"
GEN_DIR = POOL / "generated"
VAL_DIR = POOL / "validated"
REJ_DIR = POOL / "rejected"


def _ensure_dirs() -> None:
    for st in set(SUB_DIR_FOR_KIND.values()):
        (GEN_DIR / st).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / st).mkdir(parents=True, exist_ok=True)
    REJ_DIR.mkdir(parents=True, exist_ok=True)


def cmd_generate(args: argparse.Namespace) -> int:
    _ensure_dirs()
    catalogue = VariantCatalogue()
    gen = GENERATORS[args.sub_task]
    print(f"[gen] sub-task={args.sub_task} n={args.n} seed={args.seed}")
    out_dir = GEN_DIR / SUB_DIR_FOR_KIND[args.sub_task]
    written = 0
    for i in range(args.n):
        seed = args.seed + i
        try:
            task: TaskInstance = gen(catalogue, seed)
        except Exception as e:
            print(f"  seed={seed}  GENERATOR-FAIL: {type(e).__name__}: {e}")
            continue
        path = out_dir / f"{task.task_id}.json"
        path.write_text(task.model_dump_json(indent=2))
        written += 1
        print(f"  {task.task_id}  → {path.relative_to(Path.cwd())}")
    print(f"[gen] wrote {written} of {args.n} tasks to {out_dir.relative_to(Path.cwd())}")
    return 0 if written == args.n else 1


def _iter_generated(sub_task: str | None):
    targets = [sub_task] if sub_task else list(GENERATORS)
    for st in targets:
        d = GEN_DIR / st
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            yield st, p


def cmd_validate(args: argparse.Namespace) -> int:
    _ensure_dirs()
    catalogue = VariantCatalogue()
    base = args.base_url
    n_pass = n_fail = 0
    for st, path in _iter_generated(args.sub_task):
        raw = json.loads(path.read_text())
        try:
            task = TaskInstance.model_validate(raw)
        except Exception as e:
            print(f"  {path.name}  SCHEMA-FAIL: {e}")
            _reject(path, st, reason={"stage": "schema", "error": str(e)})
            n_fail += 1
            continue
        t0 = time.time()
        report = validate_task(task, catalogue, base_url=base)
        elapsed = time.time() - t0
        if report.passed:
            target = VAL_DIR / st / path.name
            shutil.move(str(path), str(target))
            n_pass += 1
            print(f"  ✓ {path.name}  (audited {report.recall_turns_audited} recall turn(s), {elapsed:.1f}s)")
        else:
            reason = {
                "stage": "ncp",
                "recall_turns_audited": report.recall_turns_audited,
                "transport_errors": report.transport_errors,
                "violations": [v.__dict__ for v in report.violations],
            }
            _reject(path, st, reason)
            n_fail += 1
            for v in report.violations[:3]:
                print(f"  ✗ {path.name}  {v.rule}/{v.type}  — {v.description}")
            for err in report.transport_errors:
                print(f"  ✗ {path.name}  transport: {err}")
    print(f"[validate] passed={n_pass} failed={n_fail}")
    return 0 if n_fail == 0 else 1


def _reject(src: Path, sub_task: str, reason: dict) -> None:
    rej_dir = REJ_DIR / src.stem
    rej_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(rej_dir / src.name))
    (rej_dir / "reason.json").write_text(json.dumps(reason, indent=2, default=str))


def cmd_pool_stats(args: argparse.Namespace) -> int:
    _ensure_dirs()
    print(f"{'sub-task':<10} {'generated':>9} {'validated':>9} {'rejected':>9}")
    print("-" * 42)
    totals = [0, 0, 0]
    for st in GENERATORS:
        g = len(list((GEN_DIR / st).glob("*.json")))
        v = len(list((VAL_DIR / st).glob("*.json")))
        r = len(list(REJ_DIR.glob(f"{st.lower()}_*")))
        print(f"{st:<10} {g:>9} {v:>9} {r:>9}")
        totals = [totals[0] + g, totals[1] + v, totals[2] + r]
    print("-" * 42)
    print(f"{'TOTAL':<10} {totals[0]:>9} {totals[1]:>9} {totals[2]:>9}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="tasks.cli")
    sp = ap.add_subparsers(dest="command", required=True)

    g = sp.add_parser("generate", help="generate N task instances of one sub-task")
    g.add_argument("sub_task", choices=list(GENERATORS))
    g.add_argument("--n", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    v = sp.add_parser("validate", help="NCP-validate generated tasks")
    v.add_argument("--sub-task", choices=list(GENERATORS), default=None,
                   help="restrict to one sub-task; default = all")
    v.add_argument("--base-url", default="http://localhost:3000")
    v.set_defaults(func=cmd_validate)

    s = sp.add_parser("pool-stats", help="show per-sub-task pool counts")
    s.set_defaults(func=cmd_pool_stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
