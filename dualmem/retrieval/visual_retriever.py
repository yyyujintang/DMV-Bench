"""VisualRetriever — cosine over visual embeddings.

Two query modes:
    - "text"  : the encoder must be ALIGNED (CLIP-like). The recall text is
                encoded via embed_text() into the SAME space as the stored
                images, and scored directly.
    - "image" : the caller supplies a query image (e.g., the current page's
                hero shot). Encoded via embed_image() and scored.

DINOv2/v3 don't have aligned text encoders, so "text" mode raises with
those backends; pair with TextRetriever via HybridRetriever instead.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dualmem.encoders.base import EncoderCapability, VisualEncoder
from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import RetrievalQuery


class VisualRetriever:
    name: str

    def __init__(self, encoder: VisualEncoder, query_mode: str = "text"):
        self.encoder = encoder
        self.query_mode = query_mode
        self.name = f"visual:{encoder.name}:{query_mode}"
        if query_mode == "text" and not (encoder.capability & EncoderCapability.ALIGNED):
            raise ValueError(
                f"VisualRetriever(query_mode='text') needs an aligned encoder; "
                f"{encoder.name} only supports {encoder.capability}."
            )

    def retrieve(self, query: RetrievalQuery, entries: List[MemoryEntry], k: int = 1,
                 query_image_path: Optional[str] = None) -> List[MemoryEntry]:
        if not entries:
            return []
        if self.query_mode == "text":
            q_vec = self.encoder.embed_text(query.recall_text)
        elif self.query_mode == "image":
            if not query_image_path:
                raise ValueError("VisualRetriever(query_mode='image') needs a query_image_path")
            q_vec = self.encoder.embed_image(query_image_path)
        else:
            raise ValueError(f"Unknown query_mode {self.query_mode!r}")
        scored = []
        for e in entries:
            if e.visual_embedding is None:
                scored.append((float("-inf"), e))
                continue
            scored.append((float(np.dot(q_vec, e.visual_embedding)), e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]
