import numpy as np

def _validate(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both vectors must be one-dimensional.")
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same dimension.")
    return a, b

def euclidean_distance(a, b):
    a, b = _validate(a, b)
    return float(np.linalg.norm(a - b))

def manhattan_distance(a, b):
    a, b = _validate(a, b)
    return float(np.sum(np.abs(a - b)))

def cosine_distance(a, b):
    a, b = _validate(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    similarity = float(np.dot(a, b) / (na * nb))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity

METRICS = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}
