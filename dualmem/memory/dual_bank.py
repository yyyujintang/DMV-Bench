"""DualBank — stores (image, visual_embedding) AND (caption, text_embedding).

Used by DualChannel and HYMEM. The visual encoder and text encoder are
independent and may use different families (e.g., CLIP for visual + SBERT
for text, or CLIP for both with the same backbone).
"""

from __future__ import annotations

from typing import List, Optional

from dualmem.encoders.base import VisualEncoder
from dualmem.encoders.cache import EmbedCache, get_default_cache
from dualmem.memory.entry import MemoryEntry


class DualBank:
    name = "dual"

    def __init__(self,
                 visual_encoder: Optional[VisualEncoder],
                 text_encoder: VisualEncoder,
                 verbal_mode: str = "vlm_caption"):
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.verbal_mode = verbal_mode
        self._entries: List[MemoryEntry] = []

    def reset(self):
        self._entries = []

    def encode(self, image_path: str, slug: str, *,
               encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry:
        cache = get_default_cache()
        v_emb = None
        # Skip visual embedding when no image is available (oracle-replay mode
        # passes image_path=""). Text channel still works; downstream visual
        # retrievers will see visual_embedding=None and score those entries
        # as -inf, which is the correct degradation.
        if self.visual_encoder is not None and image_path:
            v_key = EmbedCache._key_for_image(image_path)
            v_emb = cache.get_or_compute(
                v_key, self.visual_encoder.name + ":image",
                compute=lambda: self.visual_encoder.embed_image(image_path),
            )
        if self.verbal_mode == "encode_text":
            stored_caption = encode_text
        elif self.verbal_mode == "vlm_caption":
            stored_caption = caption or ""
        else:
            raise ValueError(f"Unknown verbal mode {self.verbal_mode!r}")
        t_emb = None
        if stored_caption:
            t_key = EmbedCache._key_for_text(stored_caption)
            t_emb = cache.get_or_compute(
                t_key, self.text_encoder.name + ":text",
                compute=lambda: self.text_encoder.embed_text(stored_caption),
            )
        entry = MemoryEntry(
            slug=slug,
            image_path=image_path,
            encode_text=encode_text,
            visual_embedding=v_emb,
            caption=stored_caption,
            text_embedding=t_emb,
            encoder_name=f"v={getattr(self.visual_encoder, 'name', None)};t={self.text_encoder.name}",
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
