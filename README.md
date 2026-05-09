# rag-core

Core building blocks for **RAG data** workflows and **AI agents**.

## Goals
- Provide reusable primitives for ingesting, cleaning, chunking, and evaluating RAG datasets.
- Offer agent-ready utilities (tools, memory, retrieval) that can be composed into projects.
- Keep the repo practical: scripts, examples, and references that help ship real systems.

## Repository layout (planned)
- `docs/`: notes, learning resources, and design decisions.
- `examples/`: runnable examples (RAG pipelines, agents, evaluations).
- `src/`: reusable library code (language/runtime to be decided).
- `data/`: sample datasets (small, non-sensitive).
- `scripts/`: maintenance utilities (downloaders, validators, converters).

## Getting started
### Build a vector database from internal `.txt` docs (Python)

1) Put your documents under `data/docs/` (or use a different `--input-dir`).

2) Install Python (Windows)
- Install Python 3.10+ from the official installer (recommended): `https://www.python.org/downloads/windows/`

3) Install dependencies and build the index

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_vector_db.py --input-dir data/docs --persist-dir vectorstore
```

4) Query the index

```bash
python scripts/query_vector_db.py "How does access policy work?"
```

This repository is intentionally minimal at initialization; expand `src/` and `examples/` as you add more RAG and agent modules.

## Contributing
See `CONTRIBUTING.md`.

## License
MIT License. See `LICENSE`.

