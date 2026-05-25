"""VisualBank — stores (image, visual_embedding) per entry.

Used by RawImage (no real retrieval — entries are just kept), CoMEM
(visual-similarity retrieval), and HYMEM's visual channel.
"""

from __future__ import annotations

from typing import List, Optional

from dualmem.encoders.base import VisualEncoder
from dualmem.encoders.cache import EmbedCache, get_default_cache
from dualmem.memory.entry import MemoryEntry


class VisualBank:
    name = "visual"

    def __init__(self, encoder: Optional[VisualEncoder] = None):
        """`encoder` is optional: a bank without an encoder stores entries
        without visual embeddings, which is the "raw image, no index"
        regime (used by the RawImage baseline)."""
        self.encoder = encoder
        self._entries: List[MemoryEntry] = []

    def reset(self):
        self._entries = []

    def encode(self, image_path: str, slug: str, *,
               encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry:
        emb = None
        # Oracle-replay passes image_path="" — encode the slug only.
        if self.encoder is not None and image_path:
            cache = get_default_cache()
            key = EmbedCache._key_for_image(image_path)
            emb = cache.get_or_compute(
                key, self.encoder.name + ":image",
                compute=lambda: self.encoder.embed_image(image_path),
            )
        entry = MemoryEntry(
            slug=slug,
            image_path=image_path,
            encode_text=encode_text,
            visual_embedding=emb,
            encoder_name=getattr(self.encoder, "name", ""),
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def find_by_slug(self, slug: str) -> Optional[MemoryEntry]:
        for e in self._entries:
            if e.slug == slug:
                return e
        return None
