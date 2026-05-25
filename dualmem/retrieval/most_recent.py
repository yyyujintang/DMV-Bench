"""MostRecentRetriever — degenerate baseline: returns the most recently encoded entry.

This is what a raw-image memory without a learned index defaults to: it
has nowhere to look up by query, so it just hands back whatever it stored
last. Used by RawImage to demonstrate the retrieval gap at full pipeline.
"""

from __future__ import annotations

from typing import List

from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import RetrievalQuery


class MostRecentRetriever:
    name = "most_recent"

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        if not entries:
            return []
        return entries[-k:][::-1]
