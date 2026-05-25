"""MultiSessionTask — composition of N sub-task sessions into one multi-session
agentic task (doc/multisession_design.md §C).

Each session is a TaskInstance JSON (see ../../tasks/schema/task_instance.py).
The runner (dualmem/runner.py::MultiSessionRunner) walks the sessions in order:

    1. For each session: create a fresh VLM context.
    2. Inject memory the session declares it needs (memory_contract.retrieves_from_prior).
    3. Run the inner agent loop (existing Playwright + ReAct).
    4. Memory bank persists across sessions; VLM context does not.

This module owns:
    - The MultiSessionTask Pydantic model (file-on-disk form).
    - The cumulative TSR evaluator (per-session boolean + product → task TSR).
    - Helpers to load a multi-session JSON from disk.

Single-session tasks are equivalent to a MultiSessionTask with len(sessions)==1.
Existing `run_agent_task` continues to work; the new runner subsumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class SessionRef(BaseModel):
    """One slot in a multi-session task.

    `task_id` references a TaskInstance JSON under `tasks/pool/validated/<SUB>/<task_id>.json`
    (the same pool the human-annotation surface reads). `sub_task` mirrors the
    referenced instance's sub_task for fast filtering without disk I/O.

    `retrieves_from_prior_override` and `must_carry_override` are populated
    by the multi-session composer to wire cross-session memory dependencies
    WITHOUT mutating the underlying TaskInstance JSONs (those stay
    single-session-pure for the human-annotation surface). The runner
    consults the overrides when present.
    """
    task_id: str
    sub_task: str            # "NC" / "SA" / "IC" / "VL" / "PD"
    order_index: int         # 0-based position in the parent task
    retrieves_from_prior_override: list[str] = Field(default_factory=list)
    must_carry_override: list[str] = Field(default_factory=list)
    # Cross-session bridging preamble, prepended to the first user turn of
    # this session by the runner (see runner.setup_phase). For order_index==0
    # this stays empty. For later sessions the composer fills it with a
    # natural-language reference to a prior session's anchor, e.g.
    #   "Earlier I showed you a rug I liked. Now I want a bookshelf in a similar tone — "
    # Without this, the runner-side session prompt is self-contained and the
    # agent has no in-prompt reason to query the memory pipeline; the
    # cross-session retrieves_from_prior contract goes unexercised.
    cross_session_user_preamble: str = ""


class MultiSessionTask(BaseModel):
    """A 3..5-session task composed of sub-task instances.

    Variant tags (see doc/multisession_design.md §E.2):
      short_3 / short_4 / short_5 / long_3 / long_4 / long_5 / homogeneous_*
    """
    task_id: str
    variant: str
    sessions: List[SessionRef]
    cumulative_turn_budget: int        # sum of expected turns across all sessions
    cumulative_gt: List[str] = Field(default_factory=list)
    """variant ids that must be in the wishlist by the end of the final session
    (PD-long cumulative GT — empty for short tasks where each session has its
    own URL-match GT)."""
    homogeneous: bool = False
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> "MultiSessionTask":
        if len(self.sessions) < 3:
            raise ValueError(f"{self.task_id}: <3 sessions ({len(self.sessions)})")
        if len(self.sessions) > 12:
            raise ValueError(f"{self.task_id}: >12 sessions ({len(self.sessions)})")
        for i, s in enumerate(self.sessions):
            if s.order_index != i:
                raise ValueError(
                    f"{self.task_id}: session[{i}].order_index={s.order_index} mismatched"
                )
        if self.homogeneous:
            kinds = {s.sub_task for s in self.sessions}
            if len(kinds) != 1:
                raise ValueError(f"{self.task_id}: homogeneous flag but kinds={kinds}")
        return self


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]   # …/DualMem_A
DEFAULT_POOL = REPO_ROOT / "tasks" / "pool" / "validated"
DEFAULT_MULTI_POOL = REPO_ROOT / "tasks" / "pool" / "multisession"


def load_multi_session(task_id: str, pool: Path = DEFAULT_MULTI_POOL) -> MultiSessionTask:
    path = pool / f"{task_id}.json"
    return MultiSessionTask.model_validate_json(path.read_text())


def save_multi_session(task: MultiSessionTask, pool: Path = DEFAULT_MULTI_POOL) -> Path:
    pool.mkdir(parents=True, exist_ok=True)
    out = pool / f"{task.task_id}.json"
    out.write_text(task.model_dump_json(indent=2))
    return out


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    """Outcome of running one session inside a multi-session task.

    Retrieval metrics (per multisession_design.md §G.2):
      - retrieval_top5: top-5 slugs the retriever returned at session start
      - irr_at_5: |top5 ∩ retrieves_from_prior| / |retrieves_from_prior|
      - mrr_at_5: 1 / (1-indexed rank of first hit within top-5); 0 if none
      - retrieval_set_hit: any top-5 slug ∈ retrieves_from_prior (boolean)
    All four are None when retrieves_from_prior is empty (session 0, or
    sessions whose memory_contract declares no cross-session dependency).
    """
    task_id: str                 # the SUB-task instance id, not the parent
    sub_task: str
    order_index: int
    correct: bool                # url_match / variant_match / wishlist_predicate
    final_url: str = ""
    final_wishlist: List[str] = field(default_factory=list)
    n_steps: int = 0
    retrieval_top5: Optional[List[str]] = None
    irr_at_5: Optional[float] = None
    mrr_at_5: Optional[float] = None
    retrieval_set_hit: Optional[bool] = None
    failure_mode: str = ""


@dataclass
class MultiSessionResult:
    """Outcome of running a whole MultiSessionTask.

    Aggregate metrics (per multisession_design.md §G.1):
      - correct          : Task Success Rate boolean (all sessions + cumulative GT pass)
      - cumulative_correct : cumulative_gt achieved at end of final session
      - progress_score   : correct sessions / total sessions (MemoryArena PS)
      - sr_at_depth      : per-session correctness indexed by depth (k=1..N).
                           Used to build the SR@k decay curve.
    """
    task_id: str
    variant: str
    sessions: List[SessionResult]
    correct: bool                # all sessions correct AND cumulative GT met
    cumulative_correct: bool     # cumulative_gt achieved at end of final session
    elapsed_ms: int = 0
    notes: str = ""

    @property
    def _scored_sessions(self) -> List[SessionResult]:
        """Recall subtasks only. Setup/browse sessions are an encoding phase
        (trivially correct) and must not count toward PS / SR@depth."""
        st = [s for s in self.sessions if s.sub_task != "Setup"]
        return st or list(self.sessions)

    @property
    def per_session_tsr(self) -> List[bool]:
        return [s.correct for s in self.sessions]

    @property
    def progress_score(self) -> float:
        st = self._scored_sessions
        if not st:
            return 0.0
        return sum(1 for s in st if s.correct) / len(st)

    @property
    def sr_at_depth(self) -> List[bool]:
        """Per-recall-subtask correctness, indexed by horizon depth k=1..N.
        Setup/browse sessions are excluded — the k-th recall subtask is
        depth k. Aggregating across tasks is the caller's responsibility."""
        return [s.correct for s in self._scored_sessions]

    def asdict(self) -> dict:
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "correct": self.correct,
            "cumulative_correct": self.cumulative_correct,
            "progress_score": self.progress_score,
            "elapsed_ms": self.elapsed_ms,
            "n_sessions": len(self.sessions),
            "per_session_correct": [
                "Pass" if s.sub_task == "Setup" else s.correct
                for s in self.sessions
            ],
            "per_session_irr_at_5": [
                -1.0 if s.irr_at_5 is None else s.irr_at_5 for s in self.sessions
            ],
            "per_session_mrr_at_5": [
                -1.0 if s.mrr_at_5 is None else s.mrr_at_5 for s in self.sessions
            ],
            "per_session_set_hit": [
                -1 if s.retrieval_set_hit is None else int(s.retrieval_set_hit)
                for s in self.sessions
            ],
            "notes": self.notes,
        }


def score_session(
    task_instance,         # tasks.schema.TaskInstance
    final_url: str,
    final_wishlist: List[str],
) -> bool:
    """Per-session success criterion.

    URL-match: final_url matches ground_truth.target_url or any
               accepted_alternatives.
    Variant-match: target_variant_id appears in final_wishlist.
    Predicate-match: deferred — caller must implement, returns False here so
                     callers know to plug their own predicate eval.
    """
    crit = task_instance.success_criteria
    gt = task_instance.ground_truth
    if crit.type == "url_match":
        if not gt.target_url:
            return False
        if final_url == gt.target_url:
            return True
        return final_url in gt.accepted_alternatives
    if crit.type == "variant_match":
        if not gt.target_variant_id:
            return False
        return gt.target_variant_id in final_wishlist
    return False   # predicate_match: caller's responsibility


def score_cumulative(
    multi_task: MultiSessionTask,
    final_wishlist: List[str],
) -> bool:
    """Cumulative GT: every variant id in cumulative_gt must be in wishlist
    at the end of the final session. Empty cumulative_gt → trivially True."""
    if not multi_task.cumulative_gt:
        return True
    return all(v in final_wishlist for v in multi_task.cumulative_gt)
