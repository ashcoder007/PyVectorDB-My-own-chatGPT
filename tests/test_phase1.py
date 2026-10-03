import numpy as np
from pyvectordb.distances import (
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
)
from pyvectordb.database import VectorDB

def test_distances():
    a = np.array([1, 0], dtype=np.float32)
    b = np.array([0, 1], dtype=np.float32)
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6
    assert abs(euclidean_distance(a, b) - 2**0.5) < 1e-6
    assert abs(manhattan_distance(a, b) - 2.0) < 1e-6

def test_database_search():
    db = VectorDB(metric="cosine")
    db.insert("a", [1, 0])
    db.insert("b", [0, 1])
    result = db.search([0.9, 0.1], k=1)
    assert result[0][1].id == "a"

def test_delete():
    db = VectorDB()
    db.insert(1, [1, 2, 3])
    db.delete(1)
    assert len(db) == 0
