"""Backend source provider — call another (text-to-image) backend to
produce a source image on demand.

Pairs naturally with an edit-mode backend: e.g. Gemini t2i produces a
clean studio shot, then Qwen-Image-Edit refines it. The cost of the
t2i call is amortised via a disk cache under
`data/vismem_diag/images_history/source-cache/`, keyed by
the t2i prompt_hash + urlHash, so iterating on the edit backend
doesn't re-invoke the upstream backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..backends.base import GenerateRequest, ImageBackend

REPO = Path(__file__).resolve().parents[2]
SOURCE_CACHE_ROOT = (
    REPO / "data" / "vismem_diag" / "images_history" / "source-cache"
)


class BackendSourceProvider:
    provider_id = "backend"

    def __init__(
        self,
        backend: ImageBackend,
        t2i_prompt_fn: Callable[[dict[str, Any]], str],
        seed_fn: Callable[[str], int],
        cache_root: Path | None = None,
    ):
        if "text2image" not in backend.capabilities:
            raise ValueError(
                f"BackendSourceProvider needs a text2image backend; "
                f"{backend.backend_id} reports {sorted(backend.capabilities)}"
            )
        self.backend = backend
        self.t2i_prompt_fn = t2i_prompt_fn
        self.seed_fn = seed_fn
        self.cache_root = cache_root or SOURCE_CACHE_ROOT
        # Tag with the *backend id* so cache keys don't collide if you
        # ever swap source backends mid-flight.
        self._cache_subdir = self.cache_root / backend.backend_id
        # We don't pre-hash the prompt template here because the
        # prompt may vary per spec; the cache key is per-spec.

    def _cache_path(self, url_hash: str, prompt: str) -> Path:
        import hashlib
        prompt_tag = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return self._cache_subdir / prompt_tag / f"{url_hash}.png"

    def source_for(self, spec: dict[str, Any]) -> bytes:
        prompt = self.t2i_prompt_fn(spec)
        cache_path = self._cache_path(spec["urlHash"], prompt)
        if cache_path.exists():
            return cache_path.read_bytes()
        seed = self.seed_fn(spec["urlHash"])
        out = self.backend.generate(GenerateRequest(prompt=prompt, seed=seed))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(out)
        return out
