"""ImageOnlyInject — emits the raw stored image as the only visual context."""

from __future__ import annotations

from dualmem.injection.base import InjectionPayload, entry_url
from dualmem.memory.entry import MemoryEntry


class ImageOnlyInject:
    name = "image_only"

    def render(self, entry: MemoryEntry) -> InjectionPayload:
        url = entry_url(entry)
        loc = f" — this memory is the product at {url}" if url else ""
        return InjectionPayload(
            text=f"Memory image (visual channel, raw){loc}:",
            images=[entry.image_path],
        )
