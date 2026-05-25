"""Perception ceiling: pure VLM 4AFC with no memory mediation.

The VLM is shown the anchor image + 4 candidates simultaneously and must
pick the matching candidate. This is the upper bound on what any memory
system on top of this VLM could possibly achieve: if the VLM cannot
discriminate even when looking at all 5 images at once, no memory system
can recover.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from dualmem.types import Trial, CellResult


PERCEPTION_INSTRUCTION = (
    "You are given an anchor image and several candidate images. "
    "Your task is to select the candidate that is the same product as the anchor. "
    "Products may differ across candidates in color, texture, or fine pattern; "
    "the matching candidate is the one that is most visually similar to the anchor in those attributes. "
    "Reply with exactly one line `Answer: N` where N is the candidate index."
)


def run_perception(trial: Trial, vlm, system_name: str = "Perception") -> CellResult:
    t0 = time.time()
    resp = vlm.four_afc(
        anchor_image_path=trial.anchor_path,
        candidate_image_paths=trial.candidate_paths,
        instructions=PERCEPTION_INSTRUCTION,
    )
    trial_id = hashlib.sha1(f"{trial.cell.cell_id}|perception|{system_name}".encode()).hexdigest()[:12]
    return CellResult(
        cell=trial.cell,
        system=system_name,
        chosen_index=resp.chosen_index,
        correct=(resp.chosen_index == trial.correct_index),
        latency_ms=resp.latency_ms,
        raw_response=resp.text,
        trial_id=trial_id,
        extra={"correct_index": trial.correct_index, "n_candidates": len(trial.candidate_paths)},
    )
