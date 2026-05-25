"""VerbalBank — stores (caption, text_embedding) per entry.

Used by TextOnly (stores the leakage text itself) and Caption (stores a
VLM-written caption of the image).
"""

from __future__ import annotations

from typing import List, Optional

from dualmem.encoders.base import VisualEncoder
from dualmem.encoders.cache import EmbedCache, get_default_cache
from dualmem.memory.entry import MemoryEntry


class VerbalBank:
    name = "verbal"

    def __init__(self, text_encoder: VisualEncoder, mode: str = "encode_text"):
        """`mode` selects what gets stored in `caption`:
            - "encode_text" : the leakage text from the encode-side trial
            - "vlm_caption" : a VLM-generated caption (caller supplies it)
        """
        self.text_encoder = text_encoder
        self.mode = mode
        self._entries: List[MemoryEntry] = []

    def reset(self):
        self._entries = []

    def encode(self, image_path: str, slug: str, *,
               encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry:
        if self.mode == "encode_text":
            stored_caption = encode_text
        elif self.mode == "vlm_caption":
            stored_caption = caption or ""
        else:
            raise ValueError(f"Unknown verbal mode {self.mode!r}")
        emb = None
        if stored_caption:
            cache = get_default_cache()
            key = EmbedCache._key_for_text(stored_caption)
            emb = cache.get_or_compute(
                key, self.text_encoder.name + ":text",
                compute=lambda: self.text_encoder.embed_text(stored_caption),
            )
        entry = MemoryEntry(
            slug=slug,
            image_path=image_path,
            encode_text=encode_text,
            caption=stored_caption,
            text_embedding=emb,
            encoder_name=getattr(self.text_encoder, "name", ""),
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
