import numpy as np
from pyvectordb.database import VectorDB

def test_kd_tree_search():
    db = VectorDB(metric="euclidean", index_type="kd_tree")
    for key, vector in {
        "a": [0.0, 0.0],
        "b": [10.0, 10.0],
        "c": [1.0, 0.0],
        "d": [0.0, 2.0],
    }.items():
        db.insert(key, vector)

    results = db.search([0.8, 0.1], k=2)
    ids = [item.id for _, item in results]
    assert ids == ["c", "a"]

def test_kd_tree_matches_brute_force():
    rng = np.random.default_rng(42)
    vectors = rng.random((100, 8), dtype=np.float32)
    query = rng.random(8, dtype=np.float32)

    brute = VectorDB("euclidean", "brute_force")
    kd = VectorDB("euclidean", "kd_tree")

    for i, vector in enumerate(vectors):
        brute.insert(i, vector)
        kd.insert(i, vector)

    brute_ids = [item.id for _, item in brute.search(query, 10)]
    kd_ids = [item.id for _, item in kd.search(query, 10)]

    assert kd_ids == brute_ids

def test_kd_tree_delete():
    db = VectorDB("euclidean", "kd_tree")
    db.insert("a", [0, 0])
    db.insert("b", [1, 1])
    db.delete("a")

    assert len(db) == 1
    assert db.search([0, 0], 1)[0][1].id == "b"
