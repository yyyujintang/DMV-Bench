"""VWA shopping site source provider.

Reads real product photos from
`VisMem-Diag/data/vismem_diag/vwa_cache/<categorySlug>/NN.jpg` and
returns them as the edit source. The cache is populated by
`tools.scrape_vwa_products` (one-time scrape, runs offline thereafter).

Each spec's `urlHash` is hashed deterministically to a cache file
index, so the same urlHash always picks the same VWA source —
reproducible across runs.

Why this exists: Qwen-Image t2i frequently invents background props
(plinths, scaffolding, contour-pattern wallpapers) despite explicit
negative prompts. Real product photos start from clean studio
compositions, so the edit step only has to apply style/colour/detail
changes, not regenerate the whole scene.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO / "data" / "vismem_diag" / "vwa_cache"


class VWASourceProvider:
    provider_id = "vwa"

    # Per-category preferred cache index. User-curated audit
    # (2026-05-16): we pick ONE source per category and all 12 variants
    # within that category share it (Qwen-Image-Edit re-styles + re-colours
    # but inherits the source's clean studio composition).
    #
    # Defaults to 0 unless overridden here. Verify by reading
    # `VisMem-Diag/data/vismem_diag/vwa_cache/<cat>/<NN>.jpg` and
    # `manifest.txt` in that directory.
    PREFERRED_INDEX: dict[str, int] = {
        # chairs/00.jpg is "Cup Holder Coaster" — not a chair.
        # chairs/01.jpg is a beige armchair, rolled arms, wood feet,
        # pure white background. Use that.
        "chairs": 1,
        # Other 9 categories' 00.jpg are correct products.
    }

    def __init__(self, cache_root: Path | None = None):
        self.cache_root = Path(cache_root) if cache_root else CACHE_ROOT
        if not self.cache_root.exists():
            raise FileNotFoundError(
                f"VWA cache not found at {self.cache_root}. "
                f"Run `python3 -m tools.scrape_vwa_products` first."
            )
        # Pre-index available images per category — fail fast at init
        # if a category isn't populated rather than per-spec.
        self._index: dict[str, list[Path]] = {}
        for cat_dir in sorted(self.cache_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            jpgs = sorted(cat_dir.glob("*.jpg"))
            if jpgs:
                self._index[cat_dir.name] = jpgs

    def categories(self) -> list[str]:
        return list(self._index)

    def source_for(self, spec: dict[str, Any]) -> bytes:
        cat = spec["categorySlug"]
        if cat not in self._index:
            raise FileNotFoundError(
                f"VWA cache has no images for category {cat!r}. "
                f"Cached: {sorted(self._index)}. "
                f"Run `python3 -m tools.scrape_vwa_products {cat}` to fix."
            )
        pool = self._index[cat]
        idx = self.PREFERRED_INDEX.get(cat, 0)
        if idx >= len(pool):
            idx = 0
        return pool[idx].read_bytes()
