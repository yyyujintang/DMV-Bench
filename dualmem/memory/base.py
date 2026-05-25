"""MemoryBank protocol — pure storage."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from dualmem.memory.entry import MemoryEntry


@runtime_checkable
class MemoryBank(Protocol):
    """A bank knows how to add entries and how to enumerate them. That's it.

    Retrievers, injectors, and ceiling runners read entries via `entries()`.
    Banks do NOT score or filter — that's retrieval's job.
    """
    name: str

    def reset(self) -> None: ...
    def encode(self, image_path: str, slug: str, *,
               encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry: ...
    def entries(self) -> List[MemoryEntry]: ...
    def find_by_slug(self, slug: str) -> Optional[MemoryEntry]: ...
