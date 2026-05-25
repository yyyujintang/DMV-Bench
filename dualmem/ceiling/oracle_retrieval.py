"""Oracle-retrieval ceiling — bypasses the retrieval layer.

We replace the system's retriever with a perfect oracle (the anchor is
injected directly). What's still in play: the *injection* format. So the
gap between this ceiling and the perception ceiling localises **injection
loss** — how much signal the format chosen by the system loses, given
perfect retrieval.

Implementation: the new MemorySystem exposes `oracle_inject(trial)`
which (a) finds the anchor entry in the bank if present, otherwise
synthesises one from the trial, then (b) calls the system's
`oracle_injector` (default = the system's normal injector). This way
oracle-retrieval is a one-line bypass at the SYSTEM level, no need to
construct a fake retriever.
"""

from __future__ import annotations

import hashlib
import time

from dualmem.types import Trial, CellResult


ORACLE_INSTRUCTION = (
    "You have memories of a product you saw before. Below is what you recall about it. "
    "Then you will be shown several candidate images and asked to identify which candidate "
    "is the same product as the one in your memory. Memory follows:"
)
RECALL_INSTRUCTION = (
    "\nGiven the memory above, which candidate is the same product? "
    "Reply with exactly one line `Answer: N` (0-indexed)."
)


def run_oracle_retrieval(trial: Trial, system, vlm) -> CellResult:
    t0 = time.time()
    payload = system.oracle_inject(trial)
    instruction = ORACLE_INSTRUCTION + "\n\n" + (payload.text or "") + RECALL_INSTRUCTION
    extra_images = payload.images or None

    resp = vlm.four_afc(
        anchor_image_path=None,
        candidate_image_paths=trial.candidate_paths,
        instructions=instruction,
        extra_context_images=extra_images,
    )
    trial_id = hashlib.sha1(f"{trial.cell.cell_id}|oracle|{system.name}".encode()).hexdigest()[:12]
    return CellResult(
        cell=trial.cell,
        system=system.name,
        chosen_index=resp.chosen_index,
        correct=(resp.chosen_index == trial.correct_index),
        latency_ms=resp.latency_ms,
        raw_response=resp.text,
        trial_id=trial_id,
        extra={"correct_index": trial.correct_index, "n_extra_images": len(extra_images or [])},
    )
