"""TextRetriever — cosine over text embeddings.

If entries lack `text_embedding`, falls back to lexical token overlap.
Used by TextOnly / Caption baselines.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

from dualmem.encoders.base import VisualEncoder
from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import RetrievalQuery


def _tokens(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class TextRetriever:
    name: str

    def __init__(self, text_encoder: VisualEncoder):
        self.text_encoder = text_encoder
        self.name = f"text:{text_encoder.name}"

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        if not entries:
            return []
        if all(e.text_embedding is None for e in entries):
            # Lexical fallback (rare — usually banks index at encode time).
            q_tok = _tokens(query.recall_text)
            scored = sorted(entries, key=lambda e: -len(q_tok & _tokens(e.caption or e.encode_text)))
            return scored[:k]
        q_vec = self.text_encoder.embed_text(query.recall_text)
        scored = []
        for e in entries:
            if e.text_embedding is None:
                scored.append((float("-inf"), e))
                continue
            scored.append((float(np.dot(q_vec, e.text_embedding)), e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]
