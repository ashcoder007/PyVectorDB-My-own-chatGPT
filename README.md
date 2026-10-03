# PyVectorDB — Phase 5

A cumulative educational Python vector database and hybrid search engine.

## Included phases

- Phase 1: vector models, distance metrics, exact brute-force search
- Phase 2: KD-Tree search
- Phase 3: HNSW approximate nearest-neighbor search
- Phase 4: SQLite persistence, CRUD, metadata, filtering, and collections
- Phase 5: BM25 keyword search and hybrid vector + keyword retrieval

## Phase 5: Hybrid search

PyVectorDB can now combine semantic vector similarity with lexical BM25 relevance. Text is read from metadata fields. By default the searchable fields are:

```text
text, content, title, description, body
```

### Keyword search

```python
from pyvectordb import VectorDB

db = VectorDB(metric="cosine")
db.insert("doc-1", [1, 0, 0], {"title": "Machine learning with Python"})

a = db.keyword_search("machine learning", k=5)
```

Results are `(bm25_score, VectorItem)` tuples sorted by descending BM25 score.

### Hybrid search

```python
results = db.hybrid_search(
    query_vector=[1, 0, 0],
    query_text="machine learning",
    k=5,
    alpha=0.7,
)
```

`alpha` controls the blend:

- `alpha=1.0`: vector retrieval only
- `alpha=0.0`: BM25 keyword retrieval only
- `alpha=0.5`: equal contribution

Hybrid scores are normalized and combined, then returned in descending relevance order.

### Custom text fields

```python
results = db.hybrid_search(
    [1, 0, 0],
    "machine learning",
    k=5,
    alpha=0.6,
    text_fields=["summary", "title"],
)
```

## Setup

From the `PyVectorDB` root folder:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The project includes both `pytest.ini` and `pyproject.toml` with `pythonpath = .`, so the local `pyvectordb` package is importable when tests are run from the project root.

## Phase 5 design notes

The BM25 implementation is intentionally dependency-free and educational. It maintains an in-memory inverted index over metadata text. Persistent databases rebuild this index from stored metadata when reopened.

The hybrid scorer converts vector distance into a similarity value, min-max normalizes vector and BM25 scores, and combines them using `alpha`. This keeps the two score types on comparable scales without requiring a shared raw score range.
