"""
TaskInstance schema (per proposal_tasks.md §2).

Pydantic v2 models for the multi-turn NCP-compliant task format. Every
generated task is validated against this schema before NCP-validator runs.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


SubTask = Literal["NC", "SA", "IC", "VL", "PD", "Setup", "NC_xs", "SA_xs", "IC_xs", "VL_xs", "PD_xs"]
Mode = Literal["encoding", "recall"]
Role = Literal["user", "agent", "system"]
ActionType = Literal["navigate", "click", "submit_query", "observe"]
FinalAction = Literal["navigate", "add_to_wishlist", "add_to_cart"]
SuccessType = Literal["url_match", "variant_match", "predicate_match"]


class AgentAction(BaseModel):
    action_type: ActionType
    target_url: Optional[str] = None
    selector: Optional[str] = None
    query: Optional[str] = None


class CueInjection(BaseModel):
    """Mirrors lib/session.ts:TaskSpecTurn.cueInjections."""
    page_id: str          # urlHash or category slug
    cue_key: str          # PeripheralCue.cueKey
    position_override: Optional[str] = None


class RecallTurnMetadata(BaseModel):
    anchor_invisibility_targets: list[str] = Field(default_factory=list)
    expected_memory_usage: Literal["positive", "negative", "both"] = "positive"


class Turn(BaseModel):
    turn_index: int
    role: Role
    mode: Mode = "encoding"

    # User turns
    content: Optional[str] = None
    is_rejection: bool = False
    references_variant: Optional[str] = None

    # Agent turns
    expected_actions: list[AgentAction] = Field(default_factory=list)
    expected_url: Optional[str] = None      # convenience for the most common "navigate" action
    cue_injections: list[CueInjection] = Field(default_factory=list)

    # Recall turn specifics
    recall_turn_metadata: Optional[RecallTurnMetadata] = None


class NcpMetadata(BaseModel):
    anchor_variant_ids: list[str]
    recall_turn_indices: list[int]
    memory_gated_branch_turn: int
    cross_turn_predicates: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    final_action: FinalAction
    target_url: Optional[str] = None
    target_variant_id: Optional[str] = None
    accepted_alternatives: list[str] = Field(default_factory=list)


class SuccessCriteria(BaseModel):
    type: SuccessType
    evaluator_fn: str
    tolerance: Optional[float] = None


class TaskMetadata(BaseModel):
    generated_by: Literal["human", "llm", "mixed"] = "llm"
    generator_seed: Optional[int] = None
    generator_version: Optional[str] = None


class MemoryContract(BaseModel):
    """Cross-session memory dependency declaration (multisession_design.md §C.3).

    `encodes` is the set of anchor variant ids this session MUST add to the
    memory bank. `retrieves_from_prior` is the set of anchor variant ids
    this session needs from earlier sessions (empty for session 1 in a
    multi-session task; populated for sessions 2..N). `must_carry_into_next`
    is the union the next session should still find in the bank after this
    one runs — used by the evaluator to score retention.

    For single-session evaluation this object can be defaulted (encodes =
    NcpMetadata.anchor_variant_ids; the rest empty); the multi-session
    runner is the consumer that wires it across sessions.
    """
    encodes: list[str] = Field(default_factory=list)
    retrieves_from_prior: list[str] = Field(default_factory=list)
    must_carry_into_next: list[str] = Field(default_factory=list)


class TaskInstance(BaseModel):
    task_id: str
    sub_task: SubTask
    grain_tier: Literal[1, 2, 3]
    category_ids: list[str]
    variants_used: list[str]
    turns: list[Turn]
    ncp_metadata: NcpMetadata
    ground_truth: GroundTruth
    success_criteria: SuccessCriteria
    memory_contract: MemoryContract = Field(default_factory=MemoryContract)
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    @model_validator(mode="after")
    def _shape_invariants(self) -> "TaskInstance":
        if len(self.turns) < 3:
            raise ValueError(f"task {self.task_id} has only {len(self.turns)} turns (need ≥3)")
        recall_idxs = self.ncp_metadata.recall_turn_indices
        if not recall_idxs:
            raise ValueError(f"task {self.task_id} declares no recall_turn_indices")
        # Every recall_turn_indices entry must point at an actual turn with mode='recall'
        turn_by_idx = {t.turn_index: t for t in self.turns}
        for idx in recall_idxs:
            t = turn_by_idx.get(idx)
            if t is None:
                raise ValueError(f"recall_turn_indices references nonexistent turn {idx}")
            if t.mode != "recall":
                raise ValueError(f"turn {idx} listed in recall_turn_indices has mode={t.mode!r}")
        # Anchor must be in variants_used
        for a in self.ncp_metadata.anchor_variant_ids:
            if a not in self.variants_used:
                raise ValueError(f"anchor {a} not in variants_used")
        # Auto-populate memory_contract.encodes from anchors if generator left
        # it empty (single-session callers shouldn't have to think about it).
        if not self.memory_contract.encodes and self.ncp_metadata.anchor_variant_ids:
            anchors = list(self.ncp_metadata.anchor_variant_ids)
            self.memory_contract = MemoryContract(
                encodes=anchors,
                retrieves_from_prior=list(self.memory_contract.retrieves_from_prior),
                must_carry_into_next=(
                    list(self.memory_contract.must_carry_into_next) or anchors
                ),
            )
        # Memory-contract: every id in encodes / retrieves_from_prior /
        # must_carry_into_next should refer to a variant the session is
        # actually aware of. retrieves_from_prior is the exception — those
        # ids come from earlier sessions and are NOT in variants_used.
        for vid in self.memory_contract.encodes:
            if vid not in self.variants_used:
                raise ValueError(f"memory_contract.encodes references {vid!r} not in variants_used")
        for vid in self.memory_contract.must_carry_into_next:
            if vid not in self.variants_used and vid not in self.memory_contract.retrieves_from_prior:
                raise ValueError(
                    f"memory_contract.must_carry_into_next references {vid!r} that is "
                    f"neither in variants_used nor in retrieves_from_prior"
                )
        return self

    def default_memory_contract(self) -> MemoryContract:
        """Heuristic contract for single-session callers: encode all anchors,
        retrieve nothing, carry all anchors forward. Generators may override."""
        anchors = list(self.ncp_metadata.anchor_variant_ids)
        return MemoryContract(
            encodes=anchors,
            retrieves_from_prior=[],
            must_carry_into_next=anchors,
        )

    def to_task_spec(self) -> dict:
        """Adapter: TaskInstance → W4 /api/session TaskSpec shape."""
        return {
            "taskId": self.task_id,
            "turns": [
                {
                    "turnIndex": t.turn_index,
                    "mode": t.mode,
                    "anchorVariantIds":
                        list(self.ncp_metadata.anchor_variant_ids) if t.mode == "recall" else [],
                    "goal": t.content or "",
                    "cueInjections": [
                        {"pageId": c.page_id, "cueKey": c.cue_key,
                         **({"positionOverride": c.position_override} if c.position_override else {})}
                        for c in t.cue_injections
                    ],
                }
                for t in self.turns
            ],
        }
