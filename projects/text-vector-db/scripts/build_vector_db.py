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

from rag_core.vector_db import index_texts


def main() -> None:
    p = argparse.ArgumentParser(description="Build a persistent Chroma vector DB from .txt files.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/docs",
        help="Directory with .txt files.",
    )
    p.add_argument(
        "--persist-dir",
        type=Path,
        default=_PROJECT_ROOT / "vectorstore",
        help="Chroma persistence directory.",
    )
    p.add_argument("--collection", default="internal_texts", help="Chroma collection name.")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name.")
    p.add_argument("--glob", default="**/*.txt", help="Glob for input files (relative to input-dir).")
    p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters.")
    p.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding/index batch size.")
    args = p.parse_args()

    index_texts(
        input_dir=args.input_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        glob_pattern=args.glob,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )

    print("Done.")
    print(f"Vector DB: {args.persist_dir}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
