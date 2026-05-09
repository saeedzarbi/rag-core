# text-vector-db

Build a **persistent** vector database from internal `.txt` documents using **Python** + **Chroma**.

## Usage

### 1) Put your internal text files here
- `data/docs/**/*.txt`

### 2) Install Python

Windows: install Python 3.10+ from `https://www.python.org/downloads/windows/`

### 3) Install deps and build the index

```bash
cd projects/text-vector-db
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_vector_db.py --input-dir data/docs --persist-dir vectorstore
```

**Imports:** Scripts add `projects/text-vector-db/src` to `sys.path`, so **`python scripts/query_vector_db.py`** runs even **without** `pip install -e .`. Prefer `pip install -e .` when you import `rag_core` from notebooks or other code.

Always install **`chromadb`** and **`sentence-transformers`** into the **same** venv you run with (`which python3` / `python3 -m pip -V` should point inside `venv/`).

Scripts default `--input-dir` / `--persist-dir` to folders under **`projects/text-vector-db/`**, even if you run them from `scripts/` (for example after `cd scripts`, `python3 query_vector_db.py "..."` still points at the project `vectorstore/`).

### 4) Query

```bash
cd projects/text-vector-db
python scripts/query_vector_db.py "How does access policy work?"
```

## Output
- The persistent DB is stored under `vectorstore/` (inside this project directory).

