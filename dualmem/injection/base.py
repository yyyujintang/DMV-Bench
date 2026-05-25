"""Injector protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

from dualmem.memory.entry import MemoryEntry


@dataclass
class InjectionPayload:
    text: str = ""
    images: List[str] = field(default_factory=list)


def entry_url(entry: MemoryEntry) -> str:
    """Best-effort site URL for a memory entry, derived from its slug
    (see dmvbench _slug_for_url). The agent needs this to navigate back to a
    remembered product — the URL is the entry's address, not its content, so
    every baseline's injector surfaces it regardless of modality."""
    s = entry.slug or ""
    if s.startswith("col_"):
        return f"/collection/{s[4:]}"
    if s.startswith("cat_"):
        return f"/category/{s[4:]}"
    if not s or s == "blank" or s.startswith("page_"):
        return ""
    return f"/product/{s}"


@runtime_checkable
class Injector(Protocol):
    name: str

    def render(self, entry: MemoryEntry) -> InjectionPayload: ...
