import numpy as np
import pytest

from pyvectordb.database import VectorDB
from pyvectordb.hybrid import BM25Index


def test_bm25_keyword_search_prefers_matching_text():
    db = VectorDB()
    db.insert("ml", [1, 0], {"title": "Machine learning with Python"})
    db.insert("db", [0, 1], {"title": "Database indexing"})

    results = db.keyword_search("machine learning", k=2)
    assert results[0][1].id == "ml"
    assert results[0][0] > 0


def test_hybrid_search_combines_vector_and_keyword():
    db = VectorDB(metric="cosine")
    db.insert("vector", [1, 0], {"title": "database systems"})
    db.insert("keyword", [0, 1], {"title": "machine learning"})

    vector_only = db.hybrid_search([1, 0], "machine learning", k=1, alpha=1.0)
    keyword_only = db.hybrid_search([1, 0], "machine learning", k=1, alpha=0.0)

    assert vector_only[0][1].id == "vector"
    assert keyword_only[0][1].id == "keyword"


def test_hybrid_metadata_filter():
    db = VectorDB(metric="cosine")
    db.insert("book", [1, 0], {"title": "machine learning", "type": "book"})
    db.insert("course", [0.99, 0.01], {"title": "machine learning", "type": "course"})

    results = db.hybrid_search(
        [1, 0],
        "machine learning",
        k=5,
        metadata_filter={"type": "course"},
    )
    assert [item.id for _, item in results] == ["course"]


def test_hybrid_custom_text_fields():
    db = VectorDB()
    db.insert("a", [1, 0], {"title": "cats", "summary": "machine learning"})
    db.insert("b", [0, 1], {"title": "machine learning", "summary": "cats"})

    results = db.keyword_search("machine learning", k=2, text_fields=["summary"])
    assert results[0][1].id == "a"

    hybrid = db.hybrid_search(
        [1, 0], "machine learning", k=1, alpha=0.5, text_fields=["summary"]
    )
    assert hybrid[0][1].id == "a"


def test_hybrid_alpha_validation():
    db = VectorDB()
    db.insert("a", [1, 0], {"title": "test"})

    with pytest.raises(ValueError):
        db.hybrid_search([1, 0], "test", alpha=-0.1)
    with pytest.raises(ValueError):
        db.hybrid_search([1, 0], "test", alpha=1.1)


def test_hybrid_persistence(tmp_path):
    path = tmp_path / "hybrid.db"
    db = VectorDB(path=path, metric="cosine")
    db.insert("a", [1, 0], {"title": "vector database"})
    db.insert("b", [0, 1], {"title": "machine learning"})
    db.close()

    reopened = VectorDB(path=path, metric="cosine")
    results = reopened.keyword_search("machine learning", k=1)
    assert results[0][1].id == "b"
    hybrid = reopened.hybrid_search([0, 1], "machine learning", k=1)
    assert hybrid[0][1].id == "b"
    reopened.close()


def test_bm25_tokenizer_and_empty_queries():
    index = BM25Index()
    index.add("a", "Python, Python databases!")
    assert index.tokenize("Python, DB!") == ["python", "db"]
    assert index.search("", 5) == []
    assert index.search("nonexistent", 5) == []
