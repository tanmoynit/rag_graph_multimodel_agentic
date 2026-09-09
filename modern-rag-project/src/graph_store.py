"""
GraphRAG layer.

At ingestion time, we use LangChain's LLMGraphTransformer to read each text
chunk and extract (entity) -[relationship]-> (entity) triples (e.g.
"Aurora Cooling Pad" -[MANUFACTURED_BY]-> "NovaTech Industries"). These are
stored as a networkx graph on disk.

At query time, we extract candidate entity names from the user's question and
pull their local neighborhood out of the graph (1-2 hops). That neighborhood
is rendered as short "subject -relation-> object" facts and injected into the
LLM's context alongside the vector-retrieved chunks. This lets the model
answer relational/logical questions ("who should handle a defect in a product
made by the same supplier as X?") that pure text similarity search struggles
with, because the answer may not share vocabulary with the question at all.
"""
import logging
from pathlib import Path
from typing import List

import networkx as nx
from langchain_core.documents import Document

from src import config
from src.llm import get_chat_llm

logger = logging.getLogger(__name__)


def build_knowledge_graph(docs: List[Document]) -> nx.MultiDiGraph:
    """Extract entities/relationships from text documents and build a graph.
    Table and image documents are skipped here (LLMGraphTransformer expects
    prose) but their content is still searchable via the vector store."""
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    text_docs = [d for d in docs if d.metadata.get("type") == "text"]
    if not text_docs:
        logger.warning("No text documents available for graph extraction.")
        return nx.MultiDiGraph()

    llm = get_chat_llm()
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(text_docs)

    graph = nx.MultiDiGraph()
    for gd in graph_documents:
        for node in gd.nodes:
            graph.add_node(node.id, type=node.type)
        for rel in gd.relationships:
            graph.add_edge(
                rel.source.id,
                rel.target.id,
                relation=rel.type,
            )

    logger.info(
        "Knowledge graph built: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def save_graph(graph: nx.MultiDiGraph, path: Path = config.GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)
    logger.info("Saved knowledge graph to %s", path)


def load_graph(path: Path = config.GRAPH_PATH) -> nx.MultiDiGraph:
    if not path.exists():
        logger.warning("No knowledge graph found at %s. Run ingest.py first.", path)
        return nx.MultiDiGraph()
    return nx.read_graphml(path)


def extract_entities_from_query(question: str) -> List[str]:
    """Ask the LLM for candidate entity names mentioned in the question, so we
    know which graph nodes to look up."""
    llm = get_chat_llm()
    prompt = (
        "Extract the key named entities (people, products, organizations, "
        "policies) mentioned or implied in this question. Return ONLY a "
        "comma-separated list, no explanation.\n\nQuestion: " + question
    )
    response = llm.invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(response.content)
    entities = [e.strip() for e in text.split(",") if e.strip()]
    return entities


def query_graph_context(graph: nx.MultiDiGraph, question: str, hops: int = 1) -> str:
    """Look up entities mentioned in the question and return their local
    neighborhood as human-readable facts, e.g. 'Aurora Cooling Pad --
    MANUFACTURED_BY --> NovaTech Industries'."""
    if graph.number_of_nodes() == 0:
        return ""

    entities = extract_entities_from_query(question)
    if not entities:
        return ""

    node_lookup = {n.lower(): n for n in graph.nodes}
    matched_nodes = set()
    for entity in entities:
        # exact + substring match, case-insensitive, since LLM-extracted
        # entity names in the query may not exactly match graph node casing
        for node_lower, node_original in node_lookup.items():
            if entity.lower() == node_lower or entity.lower() in node_lower or node_lower in entity.lower():
                matched_nodes.add(node_original)

    if not matched_nodes:
        return ""

    facts = set()
    frontier = set(matched_nodes)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for _, target, data in graph.out_edges(node, data=True):
                facts.add(f"{node} --{data.get('relation', 'RELATED_TO')}--> {target}")
                next_frontier.add(target)
            for source, _, data in graph.in_edges(node, data=True):
                facts.add(f"{source} --{data.get('relation', 'RELATED_TO')}--> {node}")
                next_frontier.add(source)
        frontier = next_frontier

    if not facts:
        return ""

    return "Knowledge graph facts:\n" + "\n".join(sorted(facts))
