"""Two-collection RAG:
  hospital-kb-chat    → แชท AI.docx only  (fast path — single embed)
  hospital-kb-general → all other files + URLs (single embed)

Pipeline: retrieve_chat → retrieve_general → web_search
"""
import glob
import logging
import os
from pathlib import Path

import chromadb
import docx
import requests
import trafilatura
from openai import OpenAI
from pypdf import PdfReader

from .config import EMBED_MODEL, CHROMA_DIR, OPENAI_TIMEOUT_S, RERANKER, COHERE_API_KEY, COHERE_RERANK_MODEL

log = logging.getLogger(__name__)
client = OpenAI()

_db = chromadb.PersistentClient(path=CHROMA_DIR)

# Cohere reranker — only initialised when RERANKER=cohere
_cohere = None
if RERANKER == "cohere" and COHERE_API_KEY:
    try:
        import cohere as _cohere_lib
        _cohere = _cohere_lib.ClientV2(api_key=COHERE_API_KEY)
        log.info("Cohere reranker enabled: %s", COHERE_RERANK_MODEL)
    except Exception as _e:
        log.warning("Cohere reranker disabled: %s", _e)

CHAT_COLLECTION    = "hospital-kb-chat"
GENERAL_COLLECTION = "hospital-kb-general"
_CHAT_FILENAME     = "แชท AI.docx"


def _load_collection(name: str):
    try:
        return _db.get_or_create_collection(name)
    except Exception:
        log.warning("Collection %s corrupt; recreating.", name)
        _db.delete_collection(name)
        return _db.create_collection(name)


_chat_col    = _load_collection(CHAT_COLLECTION)
_general_col = _load_collection(GENERAL_COLLECTION)


def embed(texts: list[str]) -> list[list[float]]:
    """Batch-embed — single API call for all texts."""
    r = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=OPENAI_TIMEOUT_S)
    return [d.embedding for d in r.data]


def _embed_batched(texts: list[str], batch: int = 512) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embed(texts[i:i + batch]))
    return out


def chunk(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    out, step = [], size - overlap
    for i in range(0, max(len(text), 1), step):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
    return out


# ---- loaders ----------------------------------------------------------------
def _data_root() -> Path:
    from .config import LOGS_DIR
    return LOGS_DIR.parent / "data"


def _load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_files():
    root = _data_root() / "raw"
    for path in glob.glob(str(root / "**" / "*"), recursive=True):
        if os.path.isdir(path):
            continue
        try:
            if path.endswith((".txt", ".md")):
                yield open(path, encoding="utf-8").read(), path
            elif path.endswith(".pdf"):
                text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
                if text.strip():
                    yield text, path
            elif path.endswith(".docx"):
                text = _load_docx(path)
                if text.strip():
                    yield text, path
        except Exception as e:
            log.warning("skip file %s: %s", path, e)


def load_urls():
    urls_path = _data_root() / "urls.txt"
    if not urls_path.exists():
        return
    for url in urls_path.read_text().splitlines():
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        try:
            html = requests.get(url, timeout=15).text
            text = trafilatura.extract(html) or ""
            if text:
                yield text, url
        except Exception as e:
            log.warning("skip url %s: %s", url, e)


# ---- ingest -----------------------------------------------------------------
def _clear(col):
    existing = col.get()
    if existing.get("ids"):
        col.delete(ids=existing["ids"])


def ingest():
    """Split แชท AI.docx into chat collection; everything else into general."""
    chat_docs, chat_metas = [], []
    gen_docs,  gen_metas  = [], []

    for text, src in load_files():
        ext   = Path(src).suffix.lstrip(".").lower()
        stype = ext if ext in ("pdf", "docx") else "md"
        for c in chunk(text):
            if Path(src).name == _CHAT_FILENAME:
                chat_docs.append(c)
                chat_metas.append({"source": src, "source_type": stype})
            else:
                gen_docs.append(c)
                gen_metas.append({"source": src, "source_type": stype})

    for text, src in load_urls():
        for c in chunk(text):
            gen_docs.append(c)
            gen_metas.append({"source": src, "source_type": "url"})

    _clear(_chat_col)
    _clear(_general_col)

    if chat_docs:
        _chat_col.add(
            ids=[str(i) for i in range(len(chat_docs))],
            documents=chat_docs,
            embeddings=_embed_batched(chat_docs),
            metadatas=chat_metas,
        )
        log.info("Chat collection: %d chunks from %s", len(chat_docs), _CHAT_FILENAME)
        print(f"Chat collection: {len(chat_docs)} chunks from {_CHAT_FILENAME}")
    else:
        log.warning("%s not found — chat collection empty", _CHAT_FILENAME)
        print(f"Warning: {_CHAT_FILENAME} not found in data/raw/ — chat collection is empty")

    if gen_docs:
        n_src = len({m["source"] for m in gen_metas})
        _general_col.add(
            ids=[str(i) for i in range(len(gen_docs))],
            documents=gen_docs,
            embeddings=_embed_batched(gen_docs),
            metadatas=gen_metas,
        )
        log.info("General collection: %d chunks from %d source(s)", len(gen_docs), n_src)
        print(f"General collection: {len(gen_docs)} chunks from {n_src} source(s)")
    else:
        log.warning("No documents for general collection.")


# ---- query-time helpers -----------------------------------------------------
def _rerank(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Rerank chunks with Cohere; falls back to original order when unavailable."""
    if not _cohere or not chunks:
        return chunks[:top_n]
    try:
        resp = _cohere.rerank(
            model=COHERE_RERANK_MODEL,
            query=question,
            documents=[c["text"] for c in chunks],
            top_n=top_n,
        )
        return [chunks[r.index] for r in resp.results]
    except Exception as e:
        log.warning("Cohere rerank failed, using original order: %s", e)
        return chunks[:top_n]


def _query_col(col, embedding: list[float], k: int) -> list[dict]:
    res   = col.query(query_embeddings=[embedding], n_results=k)
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas")  or [[]])[0]
    return [
        {
            "text":        doc,
            "source":      (m or {}).get("source",      ""),
            "source_type": (m or {}).get("source_type", ""),
        }
        for doc, m in zip(docs, metas)
    ]


def retrieve_chat(question: str, k: int = 4) -> list[dict]:
    """Embed-then-rerank retrieval from แชท AI collection."""
    e = embed([question])[0]
    candidates = _query_col(_chat_col, e, k * 3 if _cohere else k)
    return _rerank(question, candidates, k)


def retrieve_general(question: str, k: int = 5) -> list[dict]:
    """Embed-then-rerank retrieval from general (non-chat) collection."""
    e = embed([question])[0]
    candidates = _query_col(_general_col, e, k * 3 if _cohere else k)
    return _rerank(question, candidates, k)
