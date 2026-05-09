# Internal documents

Put your internal `.txt` documents in this folder (or point `--input-dir` elsewhere).

Example:

- `data/docs/policies/access_policy.txt`
- `data/docs/handbook/oncall.txt`

Build the vector database (from `projects/text-vector-db`):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_vector_db.py --input-dir data/docs --persist-dir vectorstore
```
