"""Retriever factory."""

from __future__ import annotations

from typing import Any, Callable, Dict

from dualmem.retrieval.text_retriever import TextRetriever
from dualmem.retrieval.visual_retriever import VisualRetriever
from dualmem.retrieval.hybrid_retriever import HybridRetriever
from dualmem.retrieval.oracle_retriever import OracleRetriever
from dualmem.retrieval.most_recent import MostRecentRetriever


RETRIEVER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "text":         TextRetriever,
    "visual":       VisualRetriever,
    "hybrid":       HybridRetriever,
    "oracle":       OracleRetriever,
    "most_recent":  MostRecentRetriever,
}


def make_retriever(name: str, **kwargs):
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retriever {name!r}. Known: {list(RETRIEVER_REGISTRY)}")
    return RETRIEVER_REGISTRY[name](**kwargs)
