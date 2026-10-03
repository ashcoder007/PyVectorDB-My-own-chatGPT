import heapq
from .models import VectorItem
from .distances import METRICS

class BruteForceIndex:
    def __init__(self, metric="cosine"):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric
        self.items = {}

    def insert(self, item):
        self.items[item.id] = item

    def delete(self, item_id):
        self.items.pop(item_id, None)

    def search(self, query, k=5):
        if k <= 0:
            return []
        distance_fn = METRICS[self.metric]
        scored = ((distance_fn(query, item.vector), item)
                  for item in self.items.values())
        return heapq.nsmallest(k, scored, key=lambda x: x[0])
