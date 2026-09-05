"""Incidental-Cue task schema.

A task is a LINEAR CHAIN of D sessions. There is no branching: sessions run in
order, sharing one accumulating memory bank, and the only structure sharing
between tasks is that session j depends solely on (seed, j), so a J=5 chain and
a J=15 chain built from the same seed share sessions 0..4 byte-for-byte.

Every product the agent views already carries a unique visual cue baked into
the storefront image (`images/with_cue/...`), recorded per url_hash in
`data/vismem_diag_v2/cue_registry.json`. Because the agent's browse is
autonomous, which products it will see is unknown ahead of time, so the recall
probes are NOT part of the task spec: they are filled in at run time from the
recorded encoding trajectory, one per (probe session, target session, viewed
product).

With k distinct products viewed per session (~12 for Gemini 2.5 Flash), a
D-session chain yields |cues| = k*D and |probes| = k*D(D-1)/2 under exhaustive
probe construction, or ~mc_probes*(D-1) under Monte Carlo sampling.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class SessionSpec(BaseModel):
    """One session = one short WebArena-style comparison-shopping task over a
    single category. Sessions are short; the long horizon comes from chaining
    many of them. No cue is pre-chosen -- every product the agent opens already
    carries its own."""
    session_idx: int                  # 0 .. D-1
    shopping_list: list[str]          # the ONE category browsed this session
    n_steps: int                      # session length, 22..28 steps


class RecallProbe(BaseModel):
    """An eval-only recall side-run. From the state at the end of `at_session`,
    the agent must navigate back to the product carrying `target_cue_id`.
    Read-only: scored by exact URL match, never mutates the bank."""
    probe_id: str                     # e.g. "s3_r2_p07"
    at_session: int                   # session whose end-state the probe runs from
    target_session: int               # session in which the product was viewed
    target_url_hash: str              # the recall target
    target_cue_id: str                # cue baked into that product
    target_cue_object: str
    target_cue_color: str
    reach: int                        # at_session - target_session (>= 1)


class ChainTask(BaseModel):
    """One task: a linear chain of D sessions plus its recall probes (filled at
    run time, since which products the agent views is autonomous)."""
    task_id: str
    variant: str = "ic_chain"
    n_sessions: int                   # D
    sessions: list[SessionSpec]
    probes: list[RecallProbe] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> "ChainTask":
        if len(self.sessions) != self.n_sessions:
            raise ValueError(f"{self.task_id}: {len(self.sessions)} sessions "
                             f"!= n_sessions={self.n_sessions}")
        idxs = [s.session_idx for s in self.sessions]
        if idxs != list(range(self.n_sessions)):
            raise ValueError(f"{self.task_id}: sessions must be 0..D-1 in order")
        for p in self.probes:
            if not (0 <= p.target_session < p.at_session < self.n_sessions):
                raise ValueError(f"{self.task_id}: probe {p.probe_id} bad sessions")
            if p.reach != p.at_session - p.target_session:
                raise ValueError(f"{self.task_id}: probe {p.probe_id} reach mismatch")
        return self

    def session(self, idx: int) -> SessionSpec:
        return self.sessions[idx]

    def probes_at(self, session_idx: int) -> list[RecallProbe]:
        """Probes that run from the end of `session_idx`, ordered by reach."""
        return sorted((p for p in self.probes if p.at_session == session_idx),
                      key=lambda p: p.reach)


def save_chain_task(task: ChainTask, pool: Path) -> Path:
    pool.mkdir(parents=True, exist_ok=True)
    dst = pool / f"{task.task_id}.json"
    dst.write_text(task.model_dump_json(indent=2))
    return dst


def load_chain_task(path: Path) -> ChainTask:
    return ChainTask.model_validate_json(Path(path).read_text())
