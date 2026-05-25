"""MemoryEntry — the unit of storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MemoryEntry:
    """One stored memory.

    `image_path` is always populated (memory is observation-grounded).
    `visual_embedding` / `caption` / `text_embedding` are populated by the
    bank that owns the entry — a VisualBank fills only the visual side,
    a DualBank fills both.

    `slug` is a stable identifier (the variant's catalog slug). It is used
    by oracle retrievers to identify "the anchor" without leaking it
    through other channels.
    """
    slug: str
    image_path: str
    encode_text: str = ""              # leakage text shown at encode time
    visual_embedding: Optional[np.ndarray] = None
    caption: Optional[str] = None
    text_embedding: Optional[np.ndarray] = None
    encoder_name: str = ""             # provenance for debugging
    meta: Dict[str, Any] = field(default_factory=dict)
