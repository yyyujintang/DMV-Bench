"""Memory systems — composition of (bank, retriever, injector).

A System is a THIN wrapper that:
    1. Owns one MemoryBank
    2. Owns one Retriever
    3. Owns one Injector
    4. Exposes a uniform interface for ceiling runners:
         encode(trial)
         retrieve(query) -> MemoryEntry
         inject(entry)   -> InjectionPayload
         oracle_inject(trial) -> InjectionPayload  (bypass retrieval, inject anchor)
         reset()

The six pilot baselines plus CoMEM and HYMEM all live here as small
compositions. Adding a new baseline = adding a small constructor.
"""

from dualmem.systems.base import MemorySystem
from dualmem.systems.registry import make_system, SYSTEM_REGISTRY

__all__ = ["MemorySystem", "make_system", "SYSTEM_REGISTRY"]
