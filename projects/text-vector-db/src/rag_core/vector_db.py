from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunking import chunk_text


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    document: str
    metadata: dict


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def index_texts(
    *,
    input_dir: str | Path,
    persist_dir: str | Path = "vectorstore",
    collection_name: str = "internal_texts",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    glob_pattern: str = "**/*.txt",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 64,
) -> None:
    input_dir = Path(input_dir)
    persist_dir = Path(persist_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.glob(glob_pattern) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No text files found under {input_dir} with {glob_pattern!r}")

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    def flush() -> None:
        nonlocal ids, docs, metas
        if not ids:
            return
        embeddings = model.encode(docs, show_progress_bar=False, normalize_embeddings=True)
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings.tolist())
        ids, docs, metas = [], [], []

    for file_path in tqdm(files, desc="Indexing files", unit="file"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for ch in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            doc_id = _stable_id(str(file_path.resolve()), str(ch.chunk_index), ch.text[:200])
            ids.append(doc_id)
            docs.append(ch.text)
            metas.append(
                {
                    "source": str(file_path),
                    "chunk_index": ch.chunk_index,
                }
            )
            if len(ids) >= batch_size:
                flush()

    flush()


def query_index(
    *,
    persist_dir: str | Path = "vectorstore",
    collection_name: str = "internal_texts",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    query: str,
    top_k: int = 5,
) -> list[IndexedChunk]:
    persist_dir = Path(persist_dir)
    if not persist_dir.exists() or not persist_dir.is_dir():
        raise FileNotFoundError(
            f"Vector store path is missing or not a directory: {persist_dir}. "
            "Run indexing first (e.g. scripts/build_vector_db.py) or pass the correct --persist-dir."
        )

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=collection_name)

    q_emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k, include=["documents", "metadatas"])

    ids = res.get("ids", [[]])[0]
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    out: list[IndexedChunk] = []
    for i, doc, meta in zip(ids, documents, metadatas):
        out.append(IndexedChunk(id=i, document=doc, metadata=meta))
    return out
