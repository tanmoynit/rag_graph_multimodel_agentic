"""
Agentic, modular RAG pipeline built with LangGraph.

Flow (a Corrective-RAG / Self-RAG style graph):

    retrieve --> grade_documents --+--(enough relevant docs)--> rerank --> generate --> grade_generation --+--(grounded & answers question)--> END
                                    |                                                                        |
                                    +--(not enough relevant docs, retries left)--> rewrite_query --> retrieve  |
                                                                                                                |
                                    (retries exhausted -> rerank/generate anyway, best effort) <----------------+
                                                                                (not grounded, retries left)--> rewrite_query --> retrieve

Each node is a small, independently testable function operating on a shared
typed state (`RAGState`). This modularity is what "agentic pipeline" means in
practice here: the graph dynamically decides whether to fix a bad retrieval
(via query rewriting) before ever generating an answer, and again checks
whether the generated answer is actually grounded in the retrieved evidence
before returning it to the user.
"""
import logging
from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src import config
from src.llm import get_chat_llm
from src.reranker import rerank_documents
from src.graph_store import query_graph_context

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    original_question: str
    question: str                 # possibly rewritten
    documents: List[Document]     # hybrid-retrieved candidates
    graph_context: str            # GraphRAG facts
    reranked_documents: List[Document]
    generation: str
    rewrite_count: int
    relevant_doc_count: int
    is_grounded: bool
    sources: List[dict]


def build_agent_graph(hybrid_retriever, knowledge_graph):
    """Wires up the LangGraph StateGraph. `hybrid_retriever` is a LangChain
    retriever (vector+BM25 ensemble); `knowledge_graph` is a networkx graph."""

    llm = get_chat_llm()

    # -----------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------
    def retrieve(state: RAGState) -> RAGState:
        query = state["question"]
        docs = hybrid_retriever.invoke(query)
        graph_context = query_graph_context(knowledge_graph, query)
        logger.info("Retrieved %d candidate documents for: %r", len(docs), query)
        return {"documents": docs, "graph_context": graph_context}

    def grade_documents(state: RAGState) -> RAGState:
        """Self-correction step: ask the LLM to grade each candidate chunk as
        relevant/irrelevant to the *original* question, so we can detect a bad
        retrieval before wasting a generation call on it."""
        question = state["original_question"]
        docs = state["documents"]

        relevant_docs = []
        for doc in docs:
            prompt = (
                "You are grading whether a retrieved passage is relevant to a "
                "user question. Respond with only 'yes' or 'no'.\n\n"
                f"Question: {question}\n\nPassage:\n{doc.page_content[:1000]}"
            )
            result = llm.invoke(prompt)
            verdict = (result.content if isinstance(result.content, str) else str(result.content)).strip().lower()
            if verdict.startswith("y"):
                relevant_docs.append(doc)

        logger.info("Graded documents: %d/%d relevant.", len(relevant_docs), len(docs))
        return {"documents": relevant_docs, "relevant_doc_count": len(relevant_docs)}

    def rewrite_query(state: RAGState) -> RAGState:
        """Query rewriting: reformulate the question to improve retrieval,
        e.g. resolving vague phrasing or adding likely synonyms/keywords."""
        prompt = (
            "The following question did not retrieve enough relevant search "
            "results from a knowledge base. Rewrite it to be more explicit and "
            "keyword-rich to improve retrieval, without changing its meaning. "
            "Return ONLY the rewritten question.\n\n"
            f"Original question: {state['question']}"
        )
        result = llm.invoke(prompt)
        new_question = (result.content if isinstance(result.content, str) else str(result.content)).strip()
        logger.info("Rewrote query: %r -> %r", state["question"], new_question)
        return {
            "question": new_question,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def rerank(state: RAGState) -> RAGState:
        reranked = rerank_documents(state["original_question"], state["documents"])
        return {"reranked_documents": reranked}

    def generate(state: RAGState) -> RAGState:
        docs = state.get("reranked_documents") or state.get("documents") or []
        context_blocks = [f"[{d.metadata.get('type', 'text')} | {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs]
        context = "\n\n---\n\n".join(context_blocks)

        graph_context = state.get("graph_context", "")
        full_context = context if not graph_context else f"{graph_context}\n\n---\n\n{context}"

        prompt = (
            "Answer the question using ONLY the context below. If the context "
            "does not contain the answer, say you don't have enough information. "
            "Cite sources inline like (source: filename).\n\n"
            f"Context:\n{full_context}\n\nQuestion: {state['original_question']}\n\nAnswer:"
        )
        result = llm.invoke(prompt)
        answer = result.content if isinstance(result.content, str) else str(result.content)

        sources = [
            {"source": d.metadata.get("source"), "type": d.metadata.get("type"), "page": d.metadata.get("page")}
            for d in docs
        ]
        return {"generation": answer, "sources": sources}

    def grade_generation(state: RAGState) -> RAGState:
        """Groundedness self-check: does the generated answer actually follow
        from the retrieved context, and does it address the question? This
        catches hallucination before the answer reaches the user."""
        docs = state.get("reranked_documents") or state.get("documents") or []
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "Judge the ANSWER against the CONTEXT. Respond with only 'yes' if "
            "the answer is grounded in the context (not hallucinated) AND "
            "reasonably addresses the question, otherwise respond 'no'.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['original_question']}\n\n"
            f"Answer: {state['generation']}"
        )
        result = llm.invoke(prompt)
        verdict = (result.content if isinstance(result.content, str) else str(result.content)).strip().lower()
        is_grounded = verdict.startswith("y")
        logger.info("Groundedness check: %s", is_grounded)
        return {"is_grounded": is_grounded}

    # -----------------------------------------------------------------
    # Conditional edges
    # -----------------------------------------------------------------
    def decide_after_grading(state: RAGState) -> str:
        enough_docs = state.get("relevant_doc_count", 0) >= config.MIN_RELEVANT_DOCS
        retries_left = state.get("rewrite_count", 0) < config.MAX_QUERY_REWRITES
        if enough_docs or not retries_left:
            return "rerank"
        return "rewrite_query"

    def decide_after_generation(state: RAGState) -> str:
        retries_left = state.get("rewrite_count", 0) < config.MAX_QUERY_REWRITES
        if state.get("is_grounded") or not retries_left:
            return "end"
        return "rewrite_query"

    # -----------------------------------------------------------------
    # Graph wiring
    # -----------------------------------------------------------------
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_after_grading, {"rerank": "rerank", "rewrite_query": "rewrite_query"}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation", decide_after_generation, {"end": END, "rewrite_query": "rewrite_query"}
    )

    return workflow.compile()


def ask(app, question: str) -> RAGState:
    initial_state: RAGState = {
        "original_question": question,
        "question": question,
        "rewrite_count": 0,
    }
    return app.invoke(initial_state)
