"""CaptionOnlyInject — emits the VLM-generated caption (the verbal-channel summary)."""

from __future__ import annotations

from dualmem.injection.base import InjectionPayload, entry_url
from dualmem.memory.entry import MemoryEntry


class CaptionOnlyInject:
    name = "caption_only"

    def render(self, entry: MemoryEntry) -> InjectionPayload:
        text = entry.caption or entry.encode_text or "<no caption>"
        url = entry_url(entry)
        loc = f" (product page {url})" if url else ""
        return InjectionPayload(
            text=f'Memory caption (visual channel summarised as text){loc}:\n"{text}"',
            images=[],
        )
