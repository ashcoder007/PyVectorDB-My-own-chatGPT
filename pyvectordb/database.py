from pathlib import Path
from typing import Any

from .models import VectorItem
from .brute_force import BruteForceIndex
from .kd_tree import KDTreeIndex
from .hnsw import HNSWIndex
from .storage import SQLiteStorage
from .hybrid import BM25Index


class VectorDB:
    """Vector database facade with optional SQLite persistence and metadata filtering."""

    def __init__(
        self,
        metric="cosine",
        index_type="brute_force",
        path=None,
        db_path=None,
        storage_path=None,
        collection="default",
        **index_kwargs,
    ):
        self.metric = metric
        self.index_type = index_type
        self.collection = collection
        self.storage = None
        self._bm25 = BM25Index()
        self._hybrid_text_fields = None

        persistent_path = path or db_path or storage_path
        if persistent_path is not None:
            self.storage = SQLiteStorage(persistent_path, collection)

        self.index = self._create_index(index_type, metric, index_kwargs)

        if self.storage is not None:
            for item_id, vector, metadata in self.storage.load_all():
                item = VectorItem(item_id, vector, metadata)
                self.index.insert(item)
                self._bm25.add(item.id, self._text_for_item(item))

    @staticmethod
    def _create_index(index_type, metric, index_kwargs):
        if index_type == "brute_force":
            return BruteForceIndex(metric)
        if index_type == "kd_tree":
            if metric != "euclidean":
                raise ValueError("KD-Tree currently supports Euclidean distance only.")
            return KDTreeIndex()
        if index_type == "hnsw":
            return HNSWIndex(metric=metric, **index_kwargs)
        raise ValueError(f"Unknown index type: {index_type}")

    def insert(self, item_id, vector, metadata=None, overwrite=False):
        if item_id in self.index.items and not overwrite:
            raise KeyError(f"Item already exists: {item_id}")

        item = VectorItem(item_id, vector, metadata or {})
        if item_id in self.index.items:
            self.index.delete(item_id)
        self.index.insert(item)

        if self.storage is not None:
            self.storage.upsert(item.id, item.vector, item.metadata)
        self._bm25.add(item.id, self._text_for_item(item))
        return item

    def get(self, item_id):
        return self.index.items.get(item_id)

    def update(self, item_id, vector=None, metadata=None, merge_metadata=False):
        current = self.get(item_id)
        if current is None:
            raise KeyError(f"Item not found: {item_id}")

        new_vector = current.vector if vector is None else vector
        if metadata is None:
            new_metadata = dict(current.metadata)
        elif merge_metadata:
            new_metadata = dict(current.metadata)
            new_metadata.update(metadata)
        else:
            new_metadata = dict(metadata)

        return self.insert(
            item_id,
            new_vector,
            new_metadata,
            overwrite=True,
        )

    def delete(self, item_id):
        existed = item_id in self.index.items
        self.index.delete(item_id)
        self._bm25.remove(item_id)
        if existed and self.storage is not None:
            self.storage.delete(item_id)
        return existed

    def search(self, query, k=5, metadata_filter=None):
        if k <= 0:
            return []

        if not metadata_filter:
            return self.index.search(query, k)

        # Search the complete in-memory index, then apply exact metadata filters.
        # This prioritizes correctness for filtered queries in Phase 4.
        candidates = self.index.search(query, len(self.index.items))
        return [
            result
            for result in candidates
            if self._metadata_matches(result[1].metadata, metadata_filter)
        ][:k]

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any]):
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    @staticmethod
    def _text_for_item(item: VectorItem, fields=None):
        return BM25Index.text_from_metadata(item.metadata, fields)

    def keyword_search(self, query_text, k=5, metadata_filter=None, text_fields=None):
        """Search metadata text using BM25. Results are (score, VectorItem)."""
        if k <= 0:
            return []
        bm25 = self._bm25
        if text_fields is not None:
            bm25 = BM25Index()
            for item in self.index.items.values():
                bm25.add(item.id, self._text_for_item(item, text_fields))

        ranked = bm25.search(query_text, None)
        results = []
        for score, item_id in ranked:
            item = self.get(item_id)
            if item is None:
                continue
            if metadata_filter and not self._metadata_matches(item.metadata, metadata_filter):
                continue
            results.append((score, item))
            if len(results) >= k:
                break
        return results

    def hybrid_search(
        self,
        query_vector,
        query_text,
        k=5,
        alpha=0.5,
        metadata_filter=None,
        text_fields=None,
    ):
        """Combine vector similarity and BM25 keyword relevance.

        alpha=1 uses only vector ranking; alpha=0 uses only BM25 ranking.
        Returned scores are combined relevance scores in descending order.
        """
        if k <= 0:
            return []
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if not self.index.items:
            return []

        # Build BM25 from the requested fields when a custom field set is used.
        bm25 = self._bm25
        if text_fields is not None:
            bm25 = BM25Index()
            for item in self.index.items.values():
                bm25.add(item.id, self._text_for_item(item, text_fields))

        keyword_scores = dict(bm25.search(query_text, None))
        candidates = self.index.search(query_vector, len(self.index.items))
        vector_scores = {}
        for rank, (distance, item) in enumerate(candidates):
            if metadata_filter and not self._metadata_matches(item.metadata, metadata_filter):
                continue
            vector_scores[item.id] = 1.0 / (1.0 + float(distance))

        allowed_ids = set(vector_scores)
        if metadata_filter:
            allowed_ids = {
                item_id for item_id in allowed_ids
                if self._metadata_matches(self.index.items[item_id].metadata, metadata_filter)
            }

        keyword_scores = {item_id: score for item_id, score in keyword_scores.items() if item_id in allowed_ids}
        all_ids = allowed_ids
        if not all_ids:
            return []

        vector_max = max(vector_scores[item_id] for item_id in all_ids)
        vector_min = min(vector_scores[item_id] for item_id in all_ids)
        keyword_values = [keyword_scores.get(item_id, 0.0) for item_id in all_ids]
        keyword_max = max(keyword_values) if keyword_values else 0.0

        def normalize(value, minimum, maximum):
            if maximum <= minimum:
                return 1.0 if maximum > 0 else 0.0
            return (value - minimum) / (maximum - minimum)

        scored = []
        for item_id in all_ids:
            v = normalize(vector_scores[item_id], vector_min, vector_max)
            b = normalize(keyword_scores.get(item_id, 0.0), 0.0, keyword_max)
            score = alpha * v + (1.0 - alpha) * b
            scored.append((score, self.index.items[item_id]))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].id)))
        return scored[:k]

    def count(self):
        return len(self)

    def ids(self):
        return list(self.index.items.keys())

    def collections(self):
        if self.storage is None:
            return [self.collection]
        return self.storage.collections()

    def clear(self):
        for item_id in list(self.index.items):
            self.index.delete(item_id)
        self._bm25.clear()
        if self.storage is not None:
            self.storage.clear()

    def save(self):
        if self.storage is None:
            return False
        self.storage.connection.commit()
        return True

    def close(self):
        if self.storage is not None:
            self.storage.close()
            self.storage = None

    def stats(self):
        if hasattr(self.index, "stats"):
            result = dict(self.index.stats())
        else:
            result = {"vectors": len(self)}
        result.update({
            "persistent": self.storage is not None,
            "collection": self.collection,
        })
        return result

    def __len__(self):
        return len(self.index.items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
