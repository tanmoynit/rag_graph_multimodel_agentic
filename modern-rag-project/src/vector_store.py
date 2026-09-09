"""
Chroma is used as the open-source vector database. It persists to disk under
data/chroma_db so you don't need to re-embed documents on every run.
"""
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.llm import get_embeddings

logger = logging.getLogger(__name__)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(docs: List[Document], vector_store: Chroma = None) -> Chroma:
    vector_store = vector_store or get_vector_store()
    if not docs:
        logger.warning("No documents to add to the vector store.")
        return vector_store
    # Chroma persists automatically when persist_directory is set.
    vector_store.add_documents(docs)
    logger.info("Added %d documents to Chroma collection '%s'.", len(docs), config.COLLECTION_NAME)
    return vector_store


def reset_collection() -> None:
    """Delete the current collection so the next ingest starts clean."""
    vs = get_vector_store()
    try:
        vs.delete_collection()
        logger.info("Deleted existing Chroma collection.")
    except Exception as exc:
        logger.info("No existing collection to delete (%s).", exc)
