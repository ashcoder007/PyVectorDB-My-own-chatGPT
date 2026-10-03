import numpy as np
import pytest

from pyvectordb.database import VectorDB


def test_crud_and_metadata():
    db = VectorDB()

    db.insert("a", [1, 0], {"category": "book", "year": 2026})
    db.insert("b", [0, 1], {"category": "movie", "year": 2025})

    assert db.get("a").metadata["category"] == "book"
    assert db.count() == 2
    assert set(db.ids()) == {"a", "b"}

    db.update("a", metadata={"category": "book", "year": 2027})
    assert db.get("a").metadata["year"] == 2027

    assert db.delete("b") is True
    assert db.get("b") is None
    assert db.count() == 1


def test_duplicate_insert_requires_overwrite():
    db = VectorDB()
    db.insert("a", [1, 0])

    with pytest.raises(KeyError):
        db.insert("a", [0, 1])

    db.insert("a", [0, 1], overwrite=True)
    assert np.allclose(db.get("a").vector, [0, 1])


def test_metadata_filtering():
    db = VectorDB(metric="cosine")
    db.insert("book1", [1, 0], {"type": "book", "language": "en"})
    db.insert("book2", [0.9, 0.1], {"type": "book", "language": "hi"})
    db.insert("movie1", [0.95, 0.05], {"type": "movie", "language": "en"})

    results = db.search([1, 0], k=5, metadata_filter={"type": "book"})
    ids = [item.id for _, item in results]
    assert ids == ["book1", "book2"]


def test_persistence_survives_reopen(tmp_path):
    db_path = tmp_path / "vectors.db"

    db = VectorDB(metric="cosine", path=db_path)
    db.insert("a", [1, 0, 0], {"type": "book", "title": "A"})
    db.insert("b", [0, 1, 0], {"type": "movie", "title": "B"})
    db.close()

    reopened = VectorDB(metric="cosine", path=db_path)
    assert reopened.count() == 2
    assert reopened.get("a").metadata["title"] == "A"
    assert reopened.search([1, 0, 0], 1)[0][1].id == "a"
    reopened.close()


def test_persistence_update_and_delete(tmp_path):
    db_path = tmp_path / "crud.db"
    db = VectorDB(metric="euclidean", path=db_path)
    db.insert(1, [0, 0], {"status": "old"})
    db.update(1, [1, 1], {"status": "new"})
    db.close()

    reopened = VectorDB(metric="euclidean", path=db_path)
    assert np.allclose(reopened.get(1).vector, [1, 1])
    assert reopened.get(1).metadata["status"] == "new"
    assert reopened.delete(1) is True
    reopened.close()

    final = VectorDB(metric="euclidean", path=db_path)
    assert final.count() == 0
    final.close()


def test_collection_isolation(tmp_path):
    db_path = tmp_path / "collections.db"

    books = VectorDB(path=db_path, collection="books")
    books.insert("1", [1, 0], {"type": "book"})
    books.close()

    movies = VectorDB(path=db_path, collection="movies")
    movies.insert("1", [0, 1], {"type": "movie"})
    assert movies.count() == 1
    movies.close()

    books_again = VectorDB(path=db_path, collection="books")
    assert books_again.count() == 1
    assert books_again.get("1").metadata["type"] == "book"
    assert set(books_again.collections()) == {"books", "movies"}
    books_again.close()


def test_hnsw_persistence(tmp_path):
    db_path = tmp_path / "hnsw.db"
    db = VectorDB(
        metric="cosine",
        index_type="hnsw",
        path=db_path,
        M=8,
        ef_construction=50,
        ef_search=30,
    )
    for i, vector in enumerate(([1, 0, 0], [0, 1, 0], [0.9, 0.1, 0])):
        db.insert(i, vector, {"group": "test"})
    db.close()

    reopened = VectorDB(
        metric="cosine",
        index_type="hnsw",
        path=db_path,
        M=8,
        ef_construction=50,
        ef_search=30,
    )
    assert reopened.count() == 3
    assert reopened.search([1, 0, 0], 1)[0][1].id == 0
    reopened.close()
