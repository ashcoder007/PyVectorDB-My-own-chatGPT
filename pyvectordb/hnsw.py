from dataclasses import dataclass
import heapq
import math
import random
import numpy as np
from .models import VectorItem
from .distances import METRICS

@dataclass
class _HNSWNode:
    item: VectorItem
    level: int

class HNSWIndex:
    def __init__(self, metric="cosine", M=8, ef_construction=100,
                 ef_search=50, seed=42):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if M < 2:
            raise ValueError("M must be at least 2.")
        if ef_construction < M:
            raise ValueError("ef_construction must be >= M.")
        if ef_search < 1:
            raise ValueError("ef_search must be positive.")

        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.items = {}
        self.nodes = {}
        self.graph = {}
        self.entry_point = None
        self.max_level = -1
        self.dimension = None
        self.rng = random.Random(seed)
        self.distance_fn = METRICS[metric]

    def _random_level(self):
        level = 0
        while self.rng.random() < 1.0 / math.e:
            level += 1
        return level

    def insert(self, item):
        if self.dimension is None:
            self.dimension = len(item.vector)
        elif len(item.vector) != self.dimension:
            raise ValueError("All vectors must have the same dimension.")

        if item.id in self.items:
            self.delete(item.id)

        self.items[item.id] = item
        level = self._random_level()
        self.nodes[item.id] = _HNSWNode(item, level)
        self.graph[item.id] = [set() for _ in range(level + 1)]

        if self.entry_point is None:
            self.entry_point = item.id
            self.max_level = level
            return

        current = self.entry_point

        for layer in range(self.max_level, level, -1):
            current = self._greedy_search_layer(item.vector, current, layer)

        top = min(level, self.max_level)

        for layer in range(top, -1, -1):
            candidates = self._search_layer(
                item.vector, current, self.ef_construction, layer
            )
            neighbors = self._select_neighbors(candidates, self.M)

            for neighbor_id in neighbors:
                self.graph[item.id][layer].add(neighbor_id)
                self.graph[neighbor_id][layer].add(item.id)
                self._prune(neighbor_id, layer)

            if candidates:
                current = candidates[0][1]

        if level > self.max_level:
            self.entry_point = item.id
            self.max_level = level

    def delete(self, item_id):
        if item_id not in self.items:
            return

        for neighbor_id in list(self._all_neighbors(item_id)):
            if neighbor_id in self.graph:
                for layer in range(len(self.graph[neighbor_id])):
                    self.graph[neighbor_id][layer].discard(item_id)

        self.graph.pop(item_id, None)
        self.nodes.pop(item_id, None)
        self.items.pop(item_id, None)

        if not self.items:
            self.entry_point = None
            self.max_level = -1
            self.dimension = None
            return

        self.max_level = max(n.level for n in self.nodes.values())
        candidates = [
            n for n in self.nodes.values()
            if n.level == self.max_level
        ]
        self.entry_point = candidates[0].item.id

    def _all_neighbors(self, item_id):
        result = set()
        for neighbors in self.graph.get(item_id, []):
            result.update(neighbors)
        return result

    def _greedy_search_layer(self, query, entry_id, layer):
        current = entry_id
        current_distance = self.distance_fn(
            query, self.items[current].vector
        )

        changed = True
        while changed:
            changed = False
            for neighbor in self.graph[current][layer]:
                distance = self.distance_fn(
                    query, self.items[neighbor].vector
                )
                if distance < current_distance:
                    current = neighbor
                    current_distance = distance
                    changed = True
        return current

    def _search_layer(self, query, entry_id, ef, layer):
        if entry_id not in self.graph or layer >= len(self.graph[entry_id]):
            return []

        visited = {entry_id}
        distance = self.distance_fn(
            query, self.items[entry_id].vector
        )

        candidates = [(distance, entry_id)]
        results = [(-distance, entry_id)]

        while candidates:
            current_distance, current = heapq.heappop(candidates)
            worst_distance = -results[0][0]

            if current_distance > worst_distance and len(results) >= ef:
                break

            for neighbor in self.graph[current][layer]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                d = self.distance_fn(
                    query, self.items[neighbor].vector
                )

                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))

                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted(
            [(-d, node_id) for d, node_id in results],
            key=lambda x: x[0]
        )

    def _select_neighbors(self, candidates, M):
        candidates = sorted(candidates, key=lambda x: x[0])
        return [node_id for _, node_id in candidates[:M]]

    def _prune(self, node_id, layer):
        neighbors = self.graph[node_id][layer]

        if len(neighbors) <= self.M:
            return

        scored = sorted(
            (
                self.distance_fn(
                    self.items[node_id].vector,
                    self.items[n].vector
                ),
                n
            )
            for n in neighbors
        )

        self.graph[node_id][layer] = {
            n for _, n in scored[:self.M]
        }

    def search(self, query, k=5):
        if k <= 0 or not self.items:
            return []

        query = np.asarray(query, dtype=np.float32)

        if query.ndim != 1 or len(query) != self.dimension:
            raise ValueError("Query dimension does not match the index.")

        current = self.entry_point

        for layer in range(self.max_level, 0, -1):
            current = self._greedy_search_layer(
                query, current, layer
            )

        candidates = self._search_layer(
            query,
            current,
            max(k, self.ef_search),
            0
        )

        return [
            (
                self.distance_fn(
                    query,
                    self.items[node_id].vector
                ),
                self.items[node_id]
            )
            for _, node_id in candidates[:k]
        ]

    def stats(self):
        total_edges = sum(
            len(neighbors)
            for levels in self.graph.values()
            for neighbors in levels
        )

        return {
            "vectors": len(self.items),
            "dimension": self.dimension,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "edges": total_edges,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
