"""TextOnlyInject — emits the encoded leakage text (the verbal-channel raw)."""

from __future__ import annotations

from dualmem.injection.base import InjectionPayload, entry_url
from dualmem.memory.entry import MemoryEntry


class TextOnlyInject:
    name = "text_only"

    def render(self, entry: MemoryEntry) -> InjectionPayload:
        text = entry.encode_text or entry.caption or ""
        url = entry_url(entry)
        loc = f" (product page {url})" if url else ""
        return InjectionPayload(
            text=f'Memory (verbal channel only){loc}: "{text}"',
            images=[],
        )
