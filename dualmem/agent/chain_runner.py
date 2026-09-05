"""Multi-chain driver.

Runs N chains for one memory system. The encoding trajectories are generated
once up front (system-independent -- see
`f2_encode_agent.generate_task_trajectories`) and passed in here; this driver
replays them into each system's bank and runs the recall probes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from dualmem.agent.f2_session_runner import F2TaskResult, run_f2_task


def run_f2_tasks(
    tasks: List,                            # list[ChainTask]
    system_factory: Callable[[], object],   # () -> a fresh MemorySystem
    vlm,
    *,
    trajectories_by_task: Dict[str, list],  # task_id -> list[EncodeTrajectory]
    base_url: str = "http://localhost:3000",
    playwright_page=None,
    recall_steps: int = 8,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
    mc_probes: int = 0,
    mc_seed: str = "f2.mc.v6",
) -> List[F2TaskResult]:
    """Run every chain for ONE memory system. Returns one result per chain.
    `mc_probes > 0` switches probe construction to Monte Carlo sampling
    (see `f2_session_runner.run_f2_task`)."""
    out: List[F2TaskResult] = []
    for task in tasks:
        system = system_factory()
        log_path = (log_dir / f"{task.task_id}__{system.name}.log"
                    if log_dir else None)
        out.append(run_f2_task(
            task, system, vlm,
            trajectories=trajectories_by_task[task.task_id],
            base_url=base_url, playwright_page=playwright_page,
            recall_steps=recall_steps, log_path=log_path, verbose=verbose,
            mc_probes=mc_probes, mc_seed=mc_seed))
    return out
