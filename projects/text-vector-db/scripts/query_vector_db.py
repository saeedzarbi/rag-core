from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir():
    _p = str(_SRC.resolve())
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag_core.vector_db import query_index

def main() -> None:
    p = argparse.ArgumentParser(description="Query a persistent Chroma vector DB.")
    p.add_argument(
        "--persist-dir",
        type=Path,
        default=_PROJECT_ROOT / "vectorstore",
        help="Chroma persistence directory.",
    )
    p.add_argument("--collection", default="internal_texts", help="Chroma collection name.")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name.")
    p.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    p.add_argument("query", help="Query text")
    args = p.parse_args()

    persist = args.persist_dir.expanduser().resolve()
    if not persist.is_dir():
        print(
            f"Error: vector store directory not found: {persist}\n\n"
            "Build the index first (from this project directory):\n"
            "  python3 scripts/build_vector_db.py\n\n"
            "Put .txt files under data/docs/ first. If you used a custom --persist-dir when building,\n"
            "pass the same path here: --persist-dir /path/to/your/store",
            file=sys.stderr,
        )
        sys.exit(1)

    results = query_index(
        persist_dir=persist,
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
