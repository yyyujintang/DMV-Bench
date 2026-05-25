"""ImageTextInject — Paivio-style dual coding: image + caption together."""

from __future__ import annotations

from dualmem.injection.base import InjectionPayload, entry_url
from dualmem.memory.entry import MemoryEntry


class ImageTextInject:
    name = "image_text"

    def render(self, entry: MemoryEntry) -> InjectionPayload:
        cap = entry.caption or entry.encode_text or ""
        url = entry_url(entry)
        loc = f" — product page {url}" if url else ""
        return InjectionPayload(
            text=f'Memory (dual-channel, image + caption){loc}:\nCaption: "{cap}"',
            images=[entry.image_path],
        )
