"""
Interactive CLI for the Modern RAG system.

Usage:
    python main.py
    python main.py --question "Who should handle a defect in the Aurora Cooling Pad?"
"""
import argparse
import logging

from src.vector_store import get_vector_store
from src.retrieval import load_bm25_retriever, build_hybrid_retriever
from src.graph_store import load_graph
from src.agent_graph import build_agent_graph, ask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pipeline():
    logger.info("Loading vector store, BM25 index, and knowledge graph ...")
    vector_store = get_vector_store()
    bm25_retriever = load_bm25_retriever()
    hybrid_retriever = build_hybrid_retriever(vector_store, bm25_retriever)
    knowledge_graph = load_graph()
    app = build_agent_graph(hybrid_retriever, knowledge_graph)
    return app


def print_result(result: dict):
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(result.get("generation", "(no answer generated)"))
    print("\nSOURCES:")
    for s in result.get("sources", []):
        page = f", page {s['page']}" if s.get("page") else ""
        print(f"  - {s.get('source')} [{s.get('type')}]{page}")
    print(f"\n(query rewrites used: {result.get('rewrite_count', 0)}, "
          f"grounded: {result.get('is_grounded')})")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None, help="Ask a single question and exit.")
    args = parser.parse_args()

    app = load_pipeline()

    if args.question:
        result = ask(app, args.question)
        print_result(result)
        return

    print("Modern RAG — type a question (or 'exit' to quit)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = ask(app, question)
        print_result(result)


if __name__ == "__main__":
    main()
