"""ε-margin uniqueness check on CLIP cosines.

Used by NC / SA / VL / PD ground-truth resolvers. Hard filter narrows
the candidate set; CLIP cosine ranks; top-1 wins if its margin over
top-2 ≥ ε. Otherwise the caller regenerates with a fresh seed.

Cache format: see ``tools/build_clip_cache.py``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "tasks" / "cache" / "clip_v1.json"

DEFAULT_EPSILON = 0.05


class Ambiguous(Exception):
    """Top-1 and top-2 cosines differ by < ε."""


@dataclass(frozen=True)
class ResolutionResult:
    url_hash: str
    top1_cosine: float
    top2_cosine: float
    margin: float


def _load_cache(cache_path: Path = DEFAULT_CACHE) -> dict[str, list[float]]:
    raw = json.loads(cache_path.read_text())
    return {h: v["embedding"] for h, v in raw.items()}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb)


def _centroid(vectors: list[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("centroid of empty list")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(vectors) for x in acc]


def resolve(
    candidates: Iterable[str],
    query: str | list[str],
    *,
    cache: dict[str, list[float]] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> ResolutionResult:
    """Pick the top-1 candidate by CLIP cosine to ``query``.

    Parameters
    ----------
    candidates: urlHashes after a hard filter has been applied.
    query: anchor urlHash (single-point query) or list of urlHashes
        (centroid query — score against their L2 mean).
    cache: pre-loaded {urlHash → embedding}. Lazily loaded if None.
    epsilon: margin threshold. Defaults to 0.05.

    Raises
    ------
    ValueError: candidates empty.
    Ambiguous: top-1 margin < epsilon — caller should regenerate.
    """
    if cache is None:
        cache = _load_cache()
    cand_list = list(candidates)
    if not cand_list:
        raise ValueError("resolve: candidates is empty")
    if isinstance(query, str):
        q_vec = cache[query]
    else:
        q_vec = _centroid([cache[h] for h in query])
    scored = sorted(
        ((h, cosine(cache[h], q_vec)) for h in cand_list),
        key=lambda x: x[1],
        reverse=True,
    )
    top1, top1_c = scored[0]
    top2_c = scored[1][1] if len(scored) > 1 else -1.0
    margin = top1_c - top2_c
    if len(scored) > 1 and margin < epsilon:
        raise Ambiguous(
            f"top-1 ({top1}, {top1_c:.4f}) vs top-2 "
            f"({scored[1][0]}, {top2_c:.4f}) — margin {margin:.4f} < {epsilon}"
        )
    return ResolutionResult(
        url_hash=top1,
        top1_cosine=top1_c,
        top2_cosine=top2_c,
        margin=margin,
    )


def resolve_with_retry(
    candidate_supplier: Callable[[int], list[str]],
    query_supplier: Callable[[int], str | list[str]],
    *,
    cache: dict[str, list[float]] | None = None,
    epsilon: float = DEFAULT_EPSILON,
    max_attempts: int = 20,
) -> ResolutionResult | None:
    """Convenience loop: generator-side regeneration on Ambiguous.

    ``candidate_supplier(attempt)`` and ``query_supplier(attempt)`` should
    return a fresh (filter, query) for each attempt. Returns the first
    successful ResolutionResult, or None if max_attempts is exhausted.
    """
    if cache is None:
        cache = _load_cache()
    for attempt in range(max_attempts):
        try:
            return resolve(
                candidate_supplier(attempt),
                query_supplier(attempt),
                cache=cache,
                epsilon=epsilon,
            )
        except Ambiguous:
            continue
    return None
