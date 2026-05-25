"""OracleRetriever — the bypass used by the oracle-retrieval ceiling.

Given a query that carries the anchor's slug, it returns the matching
entry from the bank directly, simulating perfect retrieval. If the anchor
is not in the bank (e.g., the test forgot to encode it), it returns the
most recent entry as a graceful fallback.
"""

from __future__ import annotations

from typing import List

from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import RetrievalQuery


class OracleRetriever:
    name = "oracle"

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        if not entries:
            return []
        if query.anchor_slug:
            for e in entries:
                if e.slug == query.anchor_slug:
                    return [e]
        return [entries[-1]]
