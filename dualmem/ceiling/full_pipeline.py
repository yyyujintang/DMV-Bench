"""Full-pipeline ceiling — all three layers active.

The system encodes the anchor PLUS K distractor memories (other cells
sampled from the inventory). At recall time, the system retrieves via
its real Retriever and injects via its real Injector. The gap from
oracle_retrieval to full_pipeline localises **retrieval loss**.
"""

from __future__ import annotations

import hashlib
import time
from typing import List

from dualmem.types import Trial, CellResult
from dualmem.retrieval.base import RetrievalQuery
from dualmem.ceiling.oracle_retrieval import ORACLE_INSTRUCTION, RECALL_INSTRUCTION


def run_full_pipeline(trial: Trial, system, vlm, distractor_memories: List[Trial]) -> CellResult:
    t0 = time.time()
    system.reset()

    # Encode anchor.
    system.encode(trial)
    # Encode distractor memories.
    for dm in distractor_memories:
        system.encode(dm)

    # Real retrieval: the query is the recall_text. The anchor slug is set
    # so an OracleRetriever (if injected for some sanity test) can identify
    # it; real retrievers ignore the slug.
    query = RetrievalQuery(recall_text=trial.recall_text, anchor_slug=trial.anchor_slug)
    retrieved = system.retrieve(query)
    if retrieved is None:
        # Should not happen in practice — bank has at least one entry.
        payload = system.oracle_inject(trial)
        retrieval_correct = False
    else:
        payload = system.inject(retrieved)
        retrieval_correct = (retrieved.slug == trial.anchor_slug)

    instruction = ORACLE_INSTRUCTION + "\n\n" + (payload.text or "") + RECALL_INSTRUCTION
    extra_images = payload.images or None

    resp = vlm.four_afc(
        anchor_image_path=None,
        candidate_image_paths=trial.candidate_paths,
        instructions=instruction,
        extra_context_images=extra_images,
    )

    trial_id = hashlib.sha1(f"{trial.cell.cell_id}|full|{system.name}".encode()).hexdigest()[:12]
    return CellResult(
        cell=trial.cell,
        system=system.name,
        chosen_index=resp.chosen_index,
        correct=(resp.chosen_index == trial.correct_index),
        latency_ms=resp.latency_ms,
        raw_response=resp.text,
        trial_id=trial_id,
        extra={
            "correct_index": trial.correct_index,
            "n_distractors": len(distractor_memories),
            "retrieval_correct": int(retrieval_correct),
        },
    )
