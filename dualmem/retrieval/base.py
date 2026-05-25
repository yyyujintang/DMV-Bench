"""Retriever protocol shared by all selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from dualmem.memory.entry import MemoryEntry


@dataclass
class RetrievalQuery:
    """Everything a retriever may use to score entries.

    - `recall_text` is the text shown to the agent at recall time (leakage text).
    - `anchor_slug` is the SLUG of the true anchor; it is REDACTED by all real
      retrievers and ONLY consumed by `OracleRetriever` to simulate a perfect
      retrieval bypass.
    """
    recall_text: str
    anchor_slug: Optional[str] = None


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        ...
