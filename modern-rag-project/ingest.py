"""
Run this once (and again whenever ingestion_data/ changes) to:
  1. Load & multimodal-parse every file in ingestion_data/ (text, PDF, images)
  2. Chunk text content
  3. Embed and store everything in the Chroma vector store
  4. Cache the raw corpus for BM25 keyword search
  5. Extract a knowledge graph (entities + relationships) for GraphRAG

Usage:
    python ingest.py
    python ingest.py --reset      # wipe existing vector store/graph first
"""
import argparse
import logging

from src import config
from src.ingestion import load_all_documents, chunk_text_documents
from src.vector_store import add_documents, reset_collection
from src.retrieval import save_bm25_corpus
from src.graph_store import build_knowledge_graph, save_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear existing vector store before ingesting.")
    parser.add_argument("--skip-graph", action="store_true", help="Skip knowledge graph extraction (faster).")
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    logger.info("Loading documents from %s ...", config.INGESTION_DIR)
    raw_docs = load_all_documents()
    if not raw_docs:
        logger.error("No documents were loaded. Add files to %s and retry.", config.INGESTION_DIR)
        return

    chunks = chunk_text_documents(raw_docs)
    by_type = {}
    for d in chunks:
        by_type[d.metadata.get("type", "text")] = by_type.get(d.metadata.get("type", "text"), 0) + 1
    logger.info("Prepared %d chunks total: %s", len(chunks), by_type)

    logger.info("Embedding & storing in Chroma ...")
    add_documents(chunks)

    logger.info("Caching corpus for BM25 keyword search ...")
    save_bm25_corpus(chunks)

    if not args.skip_graph:
        logger.info("Extracting knowledge graph (GraphRAG) ...")
        graph = build_knowledge_graph(chunks)
        save_graph(graph)
    else:
        logger.info("Skipping knowledge graph extraction (--skip-graph).")

    logger.info("Ingestion complete. You can now run: python main.py")


if __name__ == "__main__":
    main()
