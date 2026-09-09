"""
Reranking layer.

Hybrid retrieval casts a wide net (VECTOR_TOP_K + BM25_TOP_K candidates, with
duplicates). A cross-encoder reranker then scores each (query, document) pair
jointly (rather than by cosine distance between independently-computed
embeddings), which is significantly more accurate at judging true relevance.
We keep only the top RERANK_TOP_N documents that actually get passed to the
LLM for generation, which reduces both hallucination and token cost.

Uses a local, open-source cross-encoder from sentence-transformers, so no
extra API key/cost is needed for this step even if you use OpenAI for
generation.
"""
import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from src import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


def rerank_documents(query: str, docs: List[Document], top_n: int = None) -> List[Document]:
    if not docs:
        return []
    top_n = top_n or config.RERANK_TOP_N

    # de-duplicate (EnsembleRetriever can return the same chunk from both retrievers)
    seen = set()
    unique_docs = []
    for d in docs:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    model = _get_cross_encoder()
    pairs = [(query, d.page_content) for d in unique_docs]
    scores = model.predict(pairs)

    scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in scored[:top_n]]

    logger.info(
        "Reranked %d candidates -> kept top %d.", len(unique_docs), len(top_docs)
    )
    return top_docs
