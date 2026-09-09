"""
Multimodal ingestion pipeline.

Drop files into `ingestion_data/` and run `python ingest.py`.

Supported today, dispatched automatically by file extension:
  .txt / .md         -> plain text, chunked directly
  .pdf                -> text (per page) + tables (pdfplumber, converted to
                         markdown) + embedded images (extracted with PyMuPDF
                         and captioned with a vision-language model)
  .png/.jpg/.jpeg/.webp -> standalone images, captioned with a vision-language
                         model and indexed as text

Every emitted chunk is a langchain_core.documents.Document with metadata:
  source        - file name
  type          - "text" | "table" | "image"
  page          - page number if applicable

HOW TO ADD MORE FILE TYPES
---------------------------
To support .docx, .html, .pptx, etc, add a loader function below following the
same pattern as `_load_txt` / `_load_pdf`, then register it in
`_LOADERS_BY_EXTENSION` at the bottom of this file. For richer multimodal
parsing (e.g. layout-aware PDF chunking) you can swap in the `unstructured`
library's `partition_pdf(..., extract_images_in_pdf=True)` instead of the
PyMuPDF+pdfplumber combo used here — the rest of the pipeline (chunking,
embedding, graph extraction) does not need to change.
"""
import base64
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.llm import get_vision_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision captioning (shared by PDF-embedded images and standalone images)
# ---------------------------------------------------------------------------
def _caption_image_bytes(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Send an image to a vision-language model and return a text description
    that is detailed enough to be useful for retrieval (objects, chart axes/
    values, table contents, on-image text, etc.)."""
    try:
        vision_llm = get_vision_llm()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for a search index. "
                        "If it is a chart or graph, state the type, axes, and "
                        "key values/trends. If it is a table, transcribe it as "
                        "markdown. If it contains text, transcribe the text. "
                        "Be factual and specific, not decorative."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
            ],
        }
        response = vision_llm.invoke([message])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # pragma: no cover - defensive, keeps ingestion alive
        logger.warning("Image captioning failed (%s). Skipping image content.", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-file-type loaders
# ---------------------------------------------------------------------------
def _load_txt(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "type": "text"})]


def _load_image(path: Path) -> List[Document]:
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    caption = _caption_image_bytes(image_bytes, mime)
    if not caption:
        return []
    return [
        Document(
            page_content=f"[Image: {path.name}]\n{caption}",
            metadata={"source": path.name, "type": "image"},
        )
    ]


def _load_pdf(path: Path) -> List[Document]:
    docs: List[Document] = []

    # 1) Per-page text via PyPDFLoader
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(str(path)).load()
        for p in pages:
            if p.page_content.strip():
                p.metadata = {
                    "source": path.name,
                    "type": "text",
                    "page": p.metadata.get("page", None),
                }
                docs.append(p)
    except Exception as exc:
        logger.warning("Text extraction failed for %s: %s", path.name, exc)

    # 2) Tables via pdfplumber -> markdown
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue
                    md = _table_to_markdown(table)
                    docs.append(
                        Document(
                            page_content=f"[Table from {path.name}, page {page_num + 1}]\n{md}",
                            metadata={"source": path.name, "type": "table", "page": page_num + 1},
                        )
                    )
    except Exception as exc:
        logger.warning("Table extraction failed for %s: %s", path.name, exc)

    # 3) Embedded images via PyMuPDF -> caption with vision model
    try:
        import fitz  # PyMuPDF
        pdf_doc = fitz.open(str(path))
        for page_index in range(len(pdf_doc)):
            page = pdf_doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")

                # cache to disk for debugging / manual inspection
                cache_path = config.IMAGE_CACHE_DIR / f"{path.stem}_p{page_index + 1}_{img_index}.{ext}"
                cache_path.write_bytes(image_bytes)

                mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
                caption = _caption_image_bytes(image_bytes, mime)
                if caption:
                    docs.append(
                        Document(
                            page_content=(
                                f"[Image from {path.name}, page {page_index + 1}]\n{caption}"
                            ),
                            metadata={
                                "source": path.name,
                                "type": "image",
                                "page": page_index + 1,
                            },
                        )
                    )
        pdf_doc.close()
    except Exception as exc:
        logger.warning("Image extraction failed for %s: %s", path.name, exc)

    return docs


def _table_to_markdown(table: List[List]) -> str:
    header, *rows = table
    header = [str(c) if c is not None else "" for c in header]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        row = [str(c) if c is not None else "" for c in row]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_LOADERS_BY_EXTENSION = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".png": _load_image,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".webp": _load_image,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_all_documents(ingestion_dir: Path = config.INGESTION_DIR) -> List[Document]:
    """Walk ingestion_dir and load every supported file into Documents."""
    all_docs: List[Document] = []
    files = sorted(p for p in ingestion_dir.iterdir() if p.is_file())
    if not files:
        logger.warning("No files found in %s", ingestion_dir)

    for path in files:
        loader = _LOADERS_BY_EXTENSION.get(path.suffix.lower())
        if loader is None:
            logger.info("Skipping unsupported file type: %s", path.name)
            continue
        logger.info("Loading %s ...", path.name)
        try:
            docs = loader(path)
            logger.info("  -> %d chunk(s) extracted", len(docs))
            all_docs.extend(docs)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)

    return all_docs


def chunk_text_documents(docs: List[Document]) -> List[Document]:
    """Split long "text" documents into overlapping chunks. Table and image
    documents are already atomic units and are passed through unchanged so
    a table/caption is never split mid-way."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunked: List[Document] = []
    for doc in docs:
        if doc.metadata.get("type") == "text":
            chunked.extend(splitter.split_documents([doc]))
        else:
            chunked.append(doc)
    return chunked
