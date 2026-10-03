import heapq
import numpy as np
from .distances import euclidean_distance

class _KDNode:
    __slots__ = ("item", "axis", "left", "right")

    def __init__(self, item, axis):
        self.item = item
        self.axis = axis
        self.left = None
        self.right = None

class KDTreeIndex:
    def __init__(self):
        self.root = None
        self.items = {}
        self.dimension = None

    def insert(self, item):
        if self.dimension is None:
            self.dimension = len(item.vector)
        elif len(item.vector) != self.dimension:
            raise ValueError("All vectors must have the same dimension.")
        self.items[item.id] = item
        self._rebuild()

    def delete(self, item_id):
        if item_id in self.items:
            del self.items[item_id]
            self._rebuild()

    def _rebuild(self):
        values = list(self.items.values())
        self.root = self._build(values, 0) if values else None

    def _build(self, items, depth):
        if not items:
            return None
        axis = depth % self.dimension
        items.sort(key=lambda x: float(x.vector[axis]))
        mid = len(items) // 2
        node = _KDNode(items[mid], axis)
        node.left = self._build(items[:mid], depth + 1)
        node.right = self._build(items[mid + 1:], depth + 1)
        return node

    def search(self, query, k=5):
        if k <= 0 or self.root is None:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or len(query) != self.dimension:
            raise ValueError("Query dimension does not match the index.")

        heap = []
        self._search(self.root, query, k, heap)
        return sorted(
            [(-x[0], x[2]) for x in heap],
            key=lambda x: x[0]
        )

    def _search(self, node, query, k, heap):
        if node is None:
            return

        distance = euclidean_distance(query, node.item.vector)
        entry = (-distance, id(node.item), node.item)

        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, entry)

        delta = float(query[node.axis] - node.item.vector[node.axis])
        near, far = (
            (node.left, node.right)
            if delta <= 0
            else (node.right, node.left)
        )

        self._search(near, query, k, heap)

        worst = -heap[0][0] if heap else float("inf")
        if len(heap) < k or abs(delta) < worst:
            self._search(far, query, k, heap)
