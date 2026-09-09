# Modern RAG — GraphRAG + Multimodal RAG + Agentic Pipeline

A complete, runnable Retrieval-Augmented Generation system built with
**LangChain**, **LangGraph**, and **Chroma** (open-source vector database).
It combines three "modern RAG" techniques in one pipeline:

| Technique | What it does | Where it lives |
|---|---|---|
| **GraphRAG** | Extracts entities/relationships from text into a knowledge graph, and injects graph facts (not just vector matches) into the LLM's context so it can answer multi-hop, relational questions. | `src/graph_store.py` |
| **Multimodal RAG** | Parses `.txt`, `.pdf` (text + tables + embedded images), and standalone images (`.png/.jpg`). Images/tables are captioned/transcribed by a vision-language model **at ingestion time** and indexed as searchable text. | `src/ingestion.py` |
| **Agentic / modular pipeline** | A LangGraph state machine that retrieves → grades relevance → rewrites the query and retries if retrieval was bad → reranks → generates → grades the answer for hallucination → retries if not grounded. | `src/agent_graph.py` |

It ships with a small **example dataset** (`ingestion_data/`) about a
fictional "Acme Corp": an HR/expense policy, a product catalog with supplier
relationships, and an FAQ. The data is deliberately cross-referential (e.g.
"the Aurora Cooling Pad is made by NovaTech Industries, and NovaTech-supplied
products get escalated to Vendor Relations, not IT Support") so you can see
GraphRAG answer questions that plain vector search struggles with.

---

## 1. Architecture

```
                         ┌────────────────────────┐
 ingestion_data/  ──────▶│   ingest.py (one-time)  │
 (.txt / .pdf / images)  └───────────┬─────────────┘
                                      │
                 ┌────────────────────┼─────────────────────┐
                 ▼                    ▼                      ▼
        Chroma vector store   BM25 keyword cache     Knowledge graph
        (embeddings)          (data/bm25_docs.pkl)   (GraphML, entities +
        data/chroma_db/                               relationships)
                 │                    │                      │
                 └────────────────────┴──────────┬───────────┘
                                                  ▼
                                    main.py → LangGraph agent
                                    (see flow below)
```

**Agentic pipeline (`src/agent_graph.py`)**, a Corrective-RAG / Self-RAG style
graph:

```
retrieve → grade_documents ──(enough relevant docs)──▶ rerank → generate → grade_generation ──(grounded)──▶ END
               │                                                                 │
               └──(not enough, retries left)──▶ rewrite_query ──▶ retrieve      └──(not grounded, retries left)──▶ rewrite_query
```

* **retrieve** — hybrid search: dense vector search (Chroma) + sparse
  keyword search (BM25), combined via LangChain's `EnsembleRetriever`. Also
  pulls a local neighborhood of relevant knowledge-graph facts.
* **grade_documents** — LLM judges each candidate chunk relevant/irrelevant
  (self-correction: catches bad retrieval before generating).
* **rewrite_query** — LLM reformulates the question if retrieval was weak,
  and the graph loops back to `retrieve`.
* **rerank** — a local cross-encoder (`sentence-transformers`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring the surviving candidates,
  keeping only the top N — no extra API cost.
* **generate** — LLM answers using ONLY the reranked chunks + graph facts.
* **grade_generation** — LLM checks the answer is grounded in the retrieved
  context (not hallucinated); if not, and retries remain, it loops back to
  `rewrite_query`.

---

## 2. Requirements

* Python 3.10+
* An OpenAI API key (default provider — used for chat, embeddings, and
  vision/image captioning), **or** a local [Ollama](https://ollama.com)
  install if you'd rather run fully offline/free (see §6).
* No external database server needed — Chroma runs embedded and persists to
  a local folder (`data/chroma_db/`). The knowledge graph is stored as a
  plain GraphML file (`data/knowledge_graph.graphml`), no Neo4j required.

---

## 3. Setup (any machine — Linux/macOS/Windows)

```bash
# 1. Clone / unzip the project, then cd into it
cd modern-rag-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your environment
cp .env.example .env
# then open .env and set OPENAI_API_KEY=sk-...
```

### 3a. Dependencies installed by `requirements.txt`

| Package | Purpose |
|---|---|
| `langchain`, `langchain-community`, `langchain-text-splitters` | Core orchestration, document loaders, text splitting |
| `langchain-experimental` | `LLMGraphTransformer` — extracts entities/relationships for GraphRAG |
| `langgraph` | Builds the agentic state-machine pipeline |
| `langchain-openai` | OpenAI chat / embedding / vision model wrappers |
| `langchain-ollama` | (optional) local model wrappers if `LLM_PROVIDER=ollama` |
| `langchain-chroma`, `chromadb` | Open-source embedded vector database |
| `rank_bm25` | Sparse/keyword retriever (BM25) for hybrid search |
| `sentence-transformers` | Local, open-source cross-encoder reranker (no API cost) |
| `networkx` | In-memory/on-disk knowledge graph for GraphRAG |
| `pypdf` | PDF text extraction (per page) |
| `pdfplumber` | PDF table extraction → markdown |
| `PyMuPDF` (imported as `fitz`) | PDF embedded-image extraction |
| `Pillow` | Image handling |
| `python-dotenv` | Loads `.env` |
| `tiktoken` | Tokenizer used internally by OpenAI-related components |

---

## 4. Run it

### Step 1 — Ingest the data (build the vector store + graph)

```bash
python ingest.py
```

This reads every file in `ingestion_data/`, chunks the text, embeds and
stores everything in Chroma, caches the corpus for BM25, and extracts a
knowledge graph. Re-run it any time you add/change files in
`ingestion_data/`. Useful flags:

```bash
python ingest.py --reset         # wipe the existing vector store first
python ingest.py --skip-graph    # skip graph extraction (faster iteration)
```

### Step 2 — Ask questions

```bash
python main.py
# or a single one-shot question:
python main.py --question "Who should handle a defect in the Aurora Cooling Pad?"
```

Try these to see each capability in action:

* *"Can contractors work remotely?"* — plain text retrieval.
* *"My Aurora Cooling Pad stopped working, who do I contact?"* — **GraphRAG**:
  the graph knows Aurora Cooling Pad → supplied by → NovaTech Industries →
  NovaTech products → escalate to → Vendor Relations, a chain a pure
  vector/keyword match on "cooling pad" would likely miss.
* *"What products are supplied by NovaTech Industries?"* — multi-hop
  relational query, again answered from the graph, not just text similarity.

---

## 5. Multimodal ingestion — adding PDFs, images, tables

`ingestion_data/` currently contains only `.txt` files so the project runs
out of the box with zero setup friction. **The ingestion code already
supports more than text** — just drop files in and re-run `python
ingest.py`:

* **`.pdf`** → `src/ingestion.py::_load_pdf` extracts three things per file:
  1. Page text via `PyPDFLoader` (`pypdf`).
  2. Tables via `pdfplumber`, converted to markdown and indexed as their own
     searchable chunks (tagged `type=table`).
  3. Embedded images via `PyMuPDF` (`fitz`), each sent to a
     vision-language model (`gpt-4o-mini` by default) which returns a
     detailed textual description — chart axes/trends, table transcription,
     on-image text, etc. — indexed as `type=image` chunks. Extracted images
     are also cached to `data/extracted_images/` for manual inspection.
* **`.png` / `.jpg` / `.jpeg` / `.webp`** → standalone images are captioned
  the same way via `_load_image` and indexed as text.
* **`.md`** → treated like `.txt`.

So: **put a PDF or PNG in `ingestion_data/` and run `python ingest.py` — no
code changes needed.** To add a brand-new file type (e.g. `.docx`, `.html`),
write a small loader function following the `_load_txt` / `_load_pdf`
pattern in `src/ingestion.py` and register its extension in
`_LOADERS_BY_EXTENSION` at the bottom of that file.

If you want richer, layout-aware PDF parsing (e.g. reading order across
multi-column pages), swap the PyMuPDF+pdfplumber combo for the
[`unstructured`](https://github.com/Unstructured-IO/unstructured) library's
`partition_pdf(..., extract_images_in_pdf=True)` — the rest of the pipeline
(chunking, embedding, graph extraction) does not need to change.

---

## 6. Configuration

Everything is controlled via `.env` (copy `.env.example`). Key options:

```bash
LLM_PROVIDER=openai        # or "ollama" for a fully local/free setup

# OpenAI (default)
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama (local) — run `ollama serve` and `ollama pull llama3.1 llava nomic-embed-text` first
# LLM_PROVIDER=ollama
# OLLAMA_CHAT_MODEL=llama3.1
# OLLAMA_VISION_MODEL=llava          # must be a vision-capable local model
# OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Retrieval/pipeline tuning (all optional, sensible defaults in
`src/config.py`): `CHUNK_SIZE`, `CHUNK_OVERLAP`, `VECTOR_TOP_K`, `BM25_TOP_K`,
`HYBRID_VECTOR_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RERANKER_MODEL`,
`RERANK_TOP_N`, `MAX_QUERY_REWRITES`, `MIN_RELEVANT_DOCS`.

**Note on Ollama:** graph extraction quality and image captioning are
noticeably better with OpenAI's models. If you run fully local, use a
vision-capable Ollama model (e.g. `llava`) for `OLLAMA_VISION_MODEL`, or
images will be skipped with a logged warning during ingestion.

---

## 7. Swapping the vector database

Chroma was chosen because it's open-source, embedded (no server to run), and
persists to a local folder — ideal for "works on any machine" out of the
box. To swap in **Pinecone** (or Weaviate, Qdrant, Milvus, pgvector, etc.),
you only need to touch `src/vector_store.py`:

```python
# Pinecone example
from langchain_pinecone import PineconeVectorStore
def get_vector_store():
    return PineconeVectorStore(index_name=..., embedding=get_embeddings())
```

Nothing else in the pipeline (`ingest.py`, `src/retrieval.py`,
`src/agent_graph.py`) needs to change, since they only depend on the
LangChain `VectorStore`/retriever interface.

## 8. Swapping the knowledge graph store for Neo4j

`src/graph_store.py` uses `networkx` + a local GraphML file so the whole
project runs with zero infrastructure. For production scale (very large
graphs, concurrent access, Cypher queries), swap in LangChain's
`Neo4jGraph` + `LLMGraphTransformer.convert_to_graph_documents(...,
document=True)` and store directly in a running Neo4j instance instead of
`nx.write_graphml`. The extraction step (`LLMGraphTransformer`) is identical
either way.

---

## 9. Project structure

```
modern-rag-project/
├── README.md
├── requirements.txt
├── .env.example
├── ingest.py                 # one-time/whenever-data-changes: build vector store + graph
├── main.py                   # CLI: ask questions through the agentic pipeline
├── ingestion_data/            # <-- put your .txt / .pdf / .png / .jpg files here
│   ├── company_policy.txt
│   ├── product_catalog.txt
│   └── faq.txt
├── data/                      # generated at runtime (safe to delete to reset everything)
│   ├── chroma_db/              # Chroma persistence
│   ├── knowledge_graph.graphml # GraphRAG graph
│   ├── bm25_docs.pkl            # cached corpus for BM25
│   └── extracted_images/        # images extracted from PDFs, for inspection
└── src/
    ├── config.py               # all environment/config in one place
    ├── llm.py                  # provider-agnostic chat/vision/embedding model factories
    ├── ingestion.py             # multimodal loaders (txt/pdf/tables/images) + chunking
    ├── vector_store.py          # Chroma setup
    ├── retrieval.py             # BM25 + hybrid (vector+BM25) retriever
    ├── reranker.py               # cross-encoder reranking
    ├── graph_store.py            # GraphRAG: extraction, persistence, query-time lookup
    └── agent_graph.py            # LangGraph agentic pipeline (the orchestrator)
```

---

## 10. Troubleshooting

* **`OPENAI_API_KEY is not set`** — copy `.env.example` to `.env` and fill in
  your key, or set `LLM_PROVIDER=ollama`.
* **`No BM25 corpus cache found` / `No existing collection`** — run `python
  ingest.py` before `python main.py`.
* **Ingestion seems slow** — `grade_documents` and knowledge-graph
  extraction each make one LLM call per chunk; use `--skip-graph` while
  iterating on ingestion, and a smaller/cheaper chat model
  (`OPENAI_CHAT_MODEL=gpt-4o-mini`) for day-to-day use.
* **Reset everything** — delete the `data/` folder and re-run `python
  ingest.py`.
* **PDF image extraction errors** — some PDFs have malformed embedded image

* =======================================================================================================
* Next Improvement Points
* ========================================================================================================
* data/ folder: it's currently empty except an extracted_images subfolder — GitHub won't track empty folders anyway, so you don't need to worry about deleting it manually if it's already empty.
  streams; `_load_pdf` catches and logs these per-file so ingestion of other
  files continues normally.
