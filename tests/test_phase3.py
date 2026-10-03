import numpy as np
from pyvectordb.database import VectorDB

def test_hnsw_search():
    db = VectorDB(
        metric="cosine",
        index_type="hnsw",
        M=8,
        ef_construction=50,
        ef_search=30
    )

    for i, vector in enumerate([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.9, 0.1, 0],
    ]):
        db.insert(i, vector)

    results = db.search([1, 0, 0], k=2)
    assert results[0][1].id == 0

def test_hnsw_recall_on_small_dataset():
    rng = np.random.default_rng(7)
    vectors = rng.random((200, 16), dtype=np.float32)
    query = rng.random(16, dtype=np.float32)

    brute = VectorDB("cosine", "brute_force")
    hnsw = VectorDB(
        "cosine",
        "hnsw",
        M=12,
        ef_construction=100,
        ef_search=100
    )

    for i, vector in enumerate(vectors):
        brute.insert(i, vector)
        hnsw.insert(i, vector)

    exact = {x[1].id for x in brute.search(query, 10)}
    approx = {x[1].id for x in hnsw.search(query, 10)}

    recall = len(exact & approx) / len(exact)
    assert recall >= 0.5

def test_hnsw_delete():
    db = VectorDB("euclidean", "hnsw")
    db.insert("a", [0, 0])
    db.insert("b", [1, 1])
    db.delete("a")

    assert len(db) == 1
    assert db.search([0, 0], 1)[0][1].id == "b"

def test_hnsw_stats():
    db = VectorDB("cosine", "hnsw")

    for i in range(10):
        db.insert(
            i,
            np.ones(4, dtype=np.float32) * i
        )

    stats = db.stats()

    assert stats["vectors"] == 10
    assert stats["dimension"] == 4
