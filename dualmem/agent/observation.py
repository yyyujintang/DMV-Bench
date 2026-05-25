"""Page-level observation captured by Playwright for memory.encode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PageObservation:
    """One snapshot the agent has on the screen, fed into memory.encode."""
    url: str
    screenshot_path: str
    title: str = ""
    # Slug (urlHash) parsed from the URL if it's a product page. None otherwise.
    product_url_hash: Optional[str] = None
    step_index: int = 0

    @property
    def is_product_page(self) -> bool:
        return self.product_url_hash is not None
