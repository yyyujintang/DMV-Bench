"""Preference state machine for the PD (Preference Drift) task.

Per proposal_tasks_v2.md §7.3–§7.4. Three explicit revision patterns
(no implicit hedging) update a slot-based preference store. The probe
at T13 reads `final_active_state` and resolves ground truth.

This module is data-only. The PD generator (Phase 4.5) consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SlotType = Literal[
    "style",            # collection.styleSlug filter
    "price_max",        # ProductVariant.price ≤ X
    "color",            # ProductVariant.colorName filter
    "material",         # text-only filter on Product.material
    "ornament",         # ProductVariant.incidentalDetails contains X
    "category",         # Product.categoryId
]


@dataclass
class PreferenceSlot:
    """A single typed preference. `history` is append-only; `value` is
    the *current* active value (or None if revoked)."""
    slot_type: SlotType
    value: Any
    status: Literal["active", "revoked"]
    source_turn: int
    history: list[Any] = field(default_factory=list)


PreferenceState = dict[SlotType, PreferenceSlot]


@dataclass
class Pattern1Revocation:
    """`forget X` / `take back X` — slot becomes revoked, value cleared."""
    slot_type: SlotType
    turn_index: int


@dataclass
class Pattern2Replacement:
    """`instead of X, want Y` — slot value flips, status stays active."""
    slot_type: SlotType
    old_value: Any
    new_value: Any
    turn_index: int


@dataclass
class Pattern3RangeUpdate:
    """`push budget to Y` / `tighten ornament to Y` — value updates,
    history appended."""
    slot_type: SlotType
    new_value: Any
    turn_index: int


def apply_pattern1(state: PreferenceState, op: Pattern1Revocation) -> None:
    slot = state.get(op.slot_type)
    if slot is None or slot.status != "active":
        raise ValueError(
            f"Pattern1 at turn {op.turn_index}: slot {op.slot_type} not active"
        )
    slot.history.append(slot.value)
    slot.value = None
    slot.status = "revoked"


def apply_pattern2(state: PreferenceState, op: Pattern2Replacement) -> None:
    slot = state.get(op.slot_type)
    if slot is None:
        raise ValueError(
            f"Pattern2 at turn {op.turn_index}: slot {op.slot_type} doesn't exist"
        )
    if slot.value != op.old_value:
        raise ValueError(
            f"Pattern2 at turn {op.turn_index}: slot {op.slot_type} has value "
            f"{slot.value!r}, not {op.old_value!r}"
        )
    if op.new_value == op.old_value:
        raise ValueError(
            f"Pattern2 at turn {op.turn_index}: new_value must differ from old"
        )
    slot.history.append(slot.value)
    slot.value = op.new_value
    slot.status = "active"


def apply_pattern3(state: PreferenceState, op: Pattern3RangeUpdate) -> None:
    slot = state.get(op.slot_type)
    if slot is None or slot.status != "active":
        raise ValueError(
            f"Pattern3 at turn {op.turn_index}: slot {op.slot_type} not active"
        )
    if slot.value == op.new_value:
        raise ValueError(
            f"Pattern3 at turn {op.turn_index}: new_value must differ"
        )
    slot.history.append(slot.value)
    slot.value = op.new_value


def final_active_state(state: PreferenceState) -> dict[SlotType, Any]:
    """At probe time, only the active slots' current values matter."""
    return {
        st: slot.value
        for st, slot in state.items()
        if slot.status == "active"
    }


# Explicit revision-grammar templates (per v2 §7.3). Each pattern has
# ≥ 4 surface templates so generated dialogues vary lexically.
PATTERN1_TEMPLATES: list[str] = [
    "Actually, forget what I said about {slot}.",
    "I take back the {slot} requirement.",
    "Let's drop the {slot} constraint.",
    "Scratch that — no preference on {slot}.",
]

PATTERN2_TEMPLATES: list[str] = [
    "Instead of {old}, I want {new}.",
    "Forget {old}, let's go with {new}.",
    "Actually, change {old} to {new}.",
    "I've changed my mind — {new} instead of {old}.",
]

PATTERN3_TEMPLATES: list[str] = [
    "Earlier I said {old} — push that to {new}.",
    "I said {old} but specifically {new}.",
    "Let me extend that to {new} instead.",
    "Tighten the {slot} to {new}.",
]
