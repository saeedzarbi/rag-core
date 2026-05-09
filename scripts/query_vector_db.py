from __future__ import annotations

import argparse
from pathlib import Path

from rag_core.vector_db import query_index


def main() -> None:
    p = argparse.ArgumentParser(description="Query a persistent Chroma vector DB.")
    p.add_argument("--persist-dir", type=Path, default=Path("vectorstore"), help="Chroma persistence directory.")
    p.add_argument("--collection", default="internal_texts", help="Chroma collection name.")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name.")
    p.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    p.add_argument("query", help="Query text")
    args = p.parse_args()

    results = query_index(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        query=args.query,
        top_k=args.top_k,
    )

    for idx, r in enumerate(results, start=1):
        source = r.metadata.get("source")
        chunk_index = r.metadata.get("chunk_index")
        print(f"\n[{idx}] source={source} chunk={chunk_index} id={r.id}")
        print(r.document[:800].rstrip())


if __name__ == "__main__":
    main()

