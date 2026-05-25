"""Retrieval layer — selects memory entries given a query.

A Retriever knows how to score memory entries against a recall query and
return the top match (or top-k). It does NOT know:
    - how entries were stored (banks own that),
    - how the retrieved entry will be injected (injection owns that),
    - what the VLM does with the prompt (perception/VLM owns that).

This is the layer the **oracle-retrieval ceiling** bypasses: by replacing
the real Retriever with `OracleRetriever`, we make retrieval perfect and
expose what the rest of the pipeline (injection) costs.
"""

from dualmem.retrieval.base import Retriever, RetrievalQuery
from dualmem.retrieval.text_retriever import TextRetriever
from dualmem.retrieval.visual_retriever import VisualRetriever
from dualmem.retrieval.hybrid_retriever import HybridRetriever
from dualmem.retrieval.oracle_retriever import OracleRetriever
from dualmem.retrieval.most_recent import MostRecentRetriever
from dualmem.retrieval.registry import make_retriever, RETRIEVER_REGISTRY

__all__ = [
    "Retriever", "RetrievalQuery",
    "TextRetriever", "VisualRetriever", "HybridRetriever",
    "OracleRetriever", "MostRecentRetriever",
    "make_retriever", "RETRIEVER_REGISTRY",
]
