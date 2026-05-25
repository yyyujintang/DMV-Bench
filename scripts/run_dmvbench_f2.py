#!/usr/bin/env python
"""DMV-Bench Family 2 — Long-Horizon Online IC Injection runner.

Generates N task spines (each a linear D-session encoding chain + recall
probes), runs them for each memory baseline, aggregates the SR-vs-reach
retention curve.

Usage (pilot):
  python scripts/run_dmvbench_f2.py --n-sessions 4 --n-tasks 1 --max-steps 25
Usage (full):
  python scripts/run_dmvbench_f2.py --n-sessions 5 --n-tasks 100 --max-steps 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "VisMem-Diag"))

from tasks.generators.f2_online_ic import generate_f2_chain
from dualmem.agent.f2_encode_agent import generate_task_trajectories
from dualmem.agent.rollout_tree import run_f2_tasks
from dualmem.captioning import make_gemini_caption_fn
from dualmem.inventory.nano_banana import NanoBanana
from dualmem.metrics_f2 import aggregate_f2
from dualmem.systems import make_system
from dualmem.vlm import make_vlm

DEFAULT_SYSTEMS = "NoMemory,TextOnly,Caption,CoMEM,DualChannel,HYMEM"
# Trajectories are baseline-independent but **VLM-dependent** (different
# backends browse differently). Override `--traj-dir` (or set
# `F2_TRAJ_DIR`) to isolate, e.g. one cache per VLM:
#     gemini → f2_trajectories/
#     qwen   → f2_trajectories_qwen/
DEFAULT_TRAJ_DIR = (Path(__file__).resolve().parents[2] / "VisMem-Diag" / "data"
                    / "vismem_diag_v2" / "f2_trajectories")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default=DEFAULT_SYSTEMS)
    p.add_argument("--vlm", default="gemini-2.5-flash")
    p.add_argument("--n-sessions", type=int, default=5)
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="explicit comma-separated spine seeds (parallel launcher "
                        "uses this; overrides --n-tasks/--seed)")
    p.add_argument("--tag", default="f2")
    p.add_argument("--base-url", default="http://localhost:3000")
    p.add_argument("--text-encoder", default="sbert")
    p.add_argument("--visual-encoder", default="clip")
    p.add_argument("--out-dir", default=None,
                   help="explicit output dir (parallel launcher uses this)")
    p.add_argument("--gen-trajectories-only", action="store_true",
                   help="only generate+cache encoding trajectories, then exit "
                        "(parallel launcher's phase 1)")
    p.add_argument("--mc-probes", type=int, default=0,
                   help="Monte Carlo sampling: probes per reach per chain "
                        "(default 0 = exhaustive k·J(J-1)/2). Use >0 (e.g. 200) "
                        "to make long-horizon J cheap — probe count becomes "
                        "linear in J instead of quadratic. Seeded deterministically "
                        "by (mc_seed, task_id, reach) so any rerun is reproducible.")
    p.add_argument("--mc-seed", type=str, default="f2.mc.v6",
                   help="seed prefix for MC sampling (logged in probe_id).")
    p.add_argument("--traj-dir", default=None,
                   help="trajectory cache dir (default = data/vismem_diag_v2/"
                        "f2_trajectories; override e.g. .../f2_trajectories_qwen "
                        "to keep VLM backends isolated). Env var F2_TRAJ_DIR "
                        "also honored.")
    args = p.parse_args()
    # Resolve trajectory cache dir from --traj-dir / env / default.
    import os as _os
    TRAJ_DIR = Path(args.traj_dir or _os.environ.get("F2_TRAJ_DIR")
                    or DEFAULT_TRAJ_DIR)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d%H%M")
        out_dir = (REPO_ROOT / "VisMem-Diag" / "exp" / "vismem_diag" / "f2"
                   / f"{stamp}_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    if args.seeds:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seed_list = [args.seed + i for i in range(args.n_tasks)]
    tasks = [generate_f2_chain(seed=s, n_sessions=args.n_sessions,
                               max_steps=args.max_steps)
             for s in seed_list]
    n_probes = sum(len(t.probes) for t in tasks)
    print(f"[f2] {len(tasks)} task spines, D={args.n_sessions}, "
          f"{n_probes} recall probes total, {len(systems)} systems", flush=True)
    print(f"[f2] out_dir: {out_dir}", flush=True)

    vlm = make_vlm(args.vlm)
    caption_fn = make_gemini_caption_fn()
    nano_banana = NanoBanana()

    pw_ctx = pw_page = None
    try:
        from playwright.sync_api import sync_playwright
        pw_ctx = sync_playwright().start()
        browser = pw_ctx.chromium.launch()
        pw_page = browser.new_context(
            viewport={"width": 1280, "height": 900}).new_page()
        print(f"[f2] Playwright ready (base_url={args.base_url})", flush=True)
    except Exception as e:
        print(f"[f2] Playwright launch failed: {e}", flush=True)

    t0 = time.time()

    # ---- encoding-trajectory generation (ONCE, baseline-independent) ----
    print(f"[f2] generating encoding trajectories -> {TRAJ_DIR}", flush=True)
    trajectories_by_task = {}
    for task in tasks:
        tlog = []
        trajectories_by_task[task.task_id] = generate_task_trajectories(
            task, vlm, nano_banana, traj_dir=TRAJ_DIR,
            base_url=args.base_url, playwright_page=pw_page, log=tlog)
    print(f"[f2] trajectories ready ({time.time() - t0:.0f}s)", flush=True)

    if args.gen_trajectories_only:
        if pw_ctx is not None:
            try:
                pw_ctx.stop()
            except Exception:
                pass
        print("[f2] --gen-trajectories-only: done.", flush=True)
        return 0

    results_by_system = {}
    for sys_name in systems:
        t_sys = time.time()
        print(f"[f2] === {sys_name} ===", flush=True)

        def _factory(name=sys_name):
            return make_system(name, text_encoder_name=args.text_encoder,
                               visual_encoder_name=args.visual_encoder,
                               vlm_caption_fn=caption_fn)

        res = run_f2_tasks(
            tasks, _factory, vlm,
            trajectories_by_task=trajectories_by_task,
            base_url=args.base_url, playwright_page=pw_page,
            log_dir=out_dir / "logs" / sys_name, verbose=True,
            mc_probes=args.mc_probes, mc_seed=args.mc_seed)
        results_by_system[sys_name] = res
        ok = sum(p.correct for tr in res for p in tr.probes)
        tot = sum(len(tr.probes) for tr in res)
        print(f"[f2] {sys_name}: {ok}/{tot} probes correct "
              f"in {time.time() - t_sys:.0f}s", flush=True)

    if pw_ctx is not None:
        try:
            pw_ctx.stop()
        except Exception:
            pass

    table = aggregate_f2(tasks, results_by_system, out_dir=out_dir)
    print()
    print(table)
    print()
    print(f"[f2] wrote f2_summary.txt / f2_summary.csv / f2_per_probe.csv → {out_dir}")
    print(f"[f2] elapsed: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
