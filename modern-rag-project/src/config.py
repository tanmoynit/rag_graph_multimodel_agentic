"""
Central configuration for the Modern RAG project.

All values can be overridden via environment variables (see .env.example).
Nothing else in the codebase should read os.environ directly — import from
here instead, so there is a single source of truth.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the project root, if present

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INGESTION_DIR = Path(os.getenv("INGESTION_DIR", PROJECT_ROOT / "ingestion_data"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma_db"))
GRAPH_PATH = Path(os.getenv("GRAPH_PATH", DATA_DIR / "knowledge_graph.graphml"))
BM25_CACHE_PATH = Path(os.getenv("BM25_CACHE_PATH", DATA_DIR / "bm25_docs.pkl"))
IMAGE_CACHE_DIR = Path(os.getenv("IMAGE_CACHE_DIR", DATA_DIR / "extracted_images"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
# "openai"  -> uses OpenAI chat + embedding + vision models (needs OPENAI_API_KEY)
# "ollama"  -> uses a local Ollama model, fully free/offline (needs Ollama running)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# Vision-capable model used to caption images/charts/scanned tables at ingestion time
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
# Note: multimodal image captioning and OpenAI-quality graph extraction work best
# with OpenAI. If you run LLM_PROVIDER=ollama, use a vision-capable local model
# (e.g. "llava") for OLLAMA_VISION_MODEL, or images will be skipped with a warning.
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Retrieval / reranking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "6"))          # dense retriever
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "6"))               # sparse retriever
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))

# ---------------------------------------------------------------------------
# Agentic pipeline
# ---------------------------------------------------------------------------
MAX_QUERY_REWRITES = int(os.getenv("MAX_QUERY_REWRITES", "2"))
MIN_RELEVANT_DOCS = int(os.getenv("MIN_RELEVANT_DOCS", "2"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "modern_rag_collection")
