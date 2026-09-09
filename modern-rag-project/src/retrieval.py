"""
Hybrid retrieval = dense vector search (Chroma) + sparse keyword search
(BM25), combined with LangChain's EnsembleRetriever. Dense search is good at
semantic/paraphrase matches; BM25 is good at exact terms like SKUs, names,
and numbers that embeddings sometimes blur together. Combining both and then
reranking (see reranker.py) consistently outperforms either alone.
"""
import logging
import pickle
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    # langchain < 1.0
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    # langchain >= 1.0 moved several retrievers into the langchain_classic
    # compatibility package. Falling back keeps this file working on either.
    from langchain_classic.retrievers import EnsembleRetriever

from src import config

logger = logging.getLogger(__name__)


def build_bm25_retriever(docs: List[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = config.BM25_TOP_K
    return retriever


def save_bm25_corpus(docs: List[Document]) -> None:
    """BM25Retriever is in-memory only, so we cache the raw corpus to disk and
    rebuild the retriever from it on startup instead of re-running ingestion."""
    with open(config.BM25_CACHE_PATH, "wb") as f:
        pickle.dump(docs, f)


def load_bm25_retriever() -> BM25Retriever:
    if not config.BM25_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No BM25 corpus cache found at {config.BM25_CACHE_PATH}. Run ingest.py first."
        )
    with open(config.BM25_CACHE_PATH, "rb") as f:
        docs = pickle.load(f)
    return build_bm25_retriever(docs)


def build_hybrid_retriever(vector_store: Chroma, bm25_retriever: BM25Retriever) -> EnsembleRetriever:
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": config.VECTOR_TOP_K})
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[config.HYBRID_VECTOR_WEIGHT, config.HYBRID_BM25_WEIGHT],
    )
