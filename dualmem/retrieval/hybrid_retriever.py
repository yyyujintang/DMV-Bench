"""HybridRetriever — weighted sum of visual + text similarities.

Used by DualChannel / HYMEM. The weight `alpha` controls the trade-off:
alpha=1 falls back to VisualRetriever, alpha=0 falls back to TextRetriever.
"""

from __future__ import annotations

from typing import List

import numpy as np

from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import RetrievalQuery
from dualmem.retrieval.text_retriever import TextRetriever
from dualmem.retrieval.visual_retriever import VisualRetriever


class HybridRetriever:
    name: str

    def __init__(self, visual: VisualRetriever, textual: TextRetriever, alpha: float = 0.5):
        self.visual = visual
        self.textual = textual
        self.alpha = alpha
        self.name = f"hybrid(a={alpha:.2f}; v={visual.name}; t={textual.name})"

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        if not entries:
            return []
        # Get per-entry scores from each component (top-k against full pool).
        n = len(entries)
        v_top = self.visual.retrieve(query, entries, k=n)
        t_top = self.textual.retrieve(query, entries, k=n)
        v_rank = {id(e): r for r, e in enumerate(v_top)}
        t_rank = {id(e): r for r, e in enumerate(t_top)}
        # Lower rank → higher score. Combine on a 1/(1+rank) basis (Reciprocal
        # Rank Fusion-style) to be robust to scale mismatches between the two
        # similarity measures.
        scored = []
        for e in entries:
            v_score = 1.0 / (1.0 + v_rank.get(id(e), n))
            t_score = 1.0 / (1.0 + t_rank.get(id(e), n))
            scored.append((self.alpha * v_score + (1 - self.alpha) * t_score, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]


class HybridNormRetriever:
    """Variant of HybridRetriever that fuses **min-max normalized raw cosine
    similarities** instead of reciprocal-rank scores. Designed to test
    whether replacing RRF with M2A-lite-style similarity normalization closes
    the cross-over gap that DualChannel showed at long horizon
    (DualChannel decays ~7 pt r1→r9 at J=10 while M2A-lite (which uses
    min-max norm) stays flat).

    Channel pipeline (per query):
        v_scores = cosine(query_text → CLIP-image-emb, every entry)
        t_scores = cosine(query_text → SBERT,           every entry)
        v_norm   = (v - min(v)) / (max(v) - min(v))
        t_norm   = (t - min(t)) / (max(t) - min(t))
        final    = alpha · v_norm + (1 - alpha) · t_norm
    """
    name: str

    def __init__(self, visual: VisualRetriever, textual: TextRetriever, alpha: float = 0.5):
        self.visual = visual
        self.textual = textual
        self.alpha = alpha
        self.name = f"hybrid_norm(a={alpha:.2f}; v={visual.name}; t={textual.name})"

    def _scores(self, query: RetrievalQuery, entries: List[MemoryEntry]):
        # CLIP / SBERT inner-product scores — read directly off the entries'
        # cached embeddings so we keep one cosine per channel per entry.
        q_v = self.visual.encoder.embed_text(query.recall_text)
        q_t = self.textual.text_encoder.embed_text(query.recall_text)
        v_scores, t_scores = [], []
        for e in entries:
            v_scores.append(float(np.dot(q_v, e.visual_embedding))
                            if e.visual_embedding is not None else float("-inf"))
            t_scores.append(float(np.dot(q_t, e.text_embedding))
                            if e.text_embedding is not None else float("-inf"))
        return v_scores, t_scores

    @staticmethod
    def _minmax(xs):
        finite = [x for x in xs if x != float("-inf")]
        if not finite:
            return [0.0] * len(xs)
        lo, hi = min(finite), max(finite)
        span = hi - lo
        if span == 0:
            return [0.0 if x == float("-inf") else 0.5 for x in xs]
        return [0.0 if x == float("-inf") else (x - lo) / span for x in xs]

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        if not entries:
            return []
        v_scores, t_scores = self._scores(query, entries)
        v_norm = self._minmax(v_scores)
        t_norm = self._minmax(t_scores)
        combined = [self.alpha * v + (1 - self.alpha) * t
                    for v, t in zip(v_norm, t_norm)]
        ranked = sorted(zip(combined, entries), key=lambda x: -x[0])
        return [e for _, e in ranked[:k]]
