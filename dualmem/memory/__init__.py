"""Memory banks — Layer 2 of the three-layer decomposition.

A MemoryBank is pure STORAGE. It knows how to encode an entry (anchor image
+ optional caption) and how to enumerate stored entries; it does NOT know
how to retrieve or how to inject. Retrieval lives in `dualmem.retrieval`,
injection in `dualmem.injection`.

Three bank types:
    - VisualBank : (image, visual_embedding) — used by RawImage / CoMEM
    - VerbalBank : (caption, text_embedding) — used by TextOnly / Caption
    - DualBank   : both channels in parallel — used by DualChannel / HYMEM
"""

from dualmem.memory.entry import MemoryEntry
from dualmem.memory.base import MemoryBank
from dualmem.memory.visual_bank import VisualBank
from dualmem.memory.verbal_bank import VerbalBank
from dualmem.memory.dual_bank import DualBank

__all__ = [
    "MemoryEntry", "MemoryBank",
    "VisualBank", "VerbalBank", "DualBank",
]
