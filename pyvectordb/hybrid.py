import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


class BM25Index:
    """Small in-memory BM25 index for metadata text fields."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0:
            raise ValueError("k1 must be >= 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")
        self.k1 = float(k1)
        self.b = float(b)
        self.documents: dict[Any, list[str]] = {}
        self.term_frequencies: dict[Any, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        self.total_length = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(str(text))]

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(
                f"{key} {BM25Index._stringify(item)}"
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(BM25Index._stringify(item) for item in value)
        return str(value)

    @classmethod
    def text_from_metadata(
        cls,
        metadata: dict[str, Any],
        fields: Iterable[str] | None = None,
    ) -> str:
        if fields is None:
            fields = ("text", "content", "title", "description", "body")
        parts = []
        for field in fields:
            if field in metadata:
                parts.append(cls._stringify(metadata[field]))
        return " ".join(parts)

    def add(self, item_id: Any, text: str):
        self.remove(item_id)
        tokens = self.tokenize(text)
        self.documents[item_id] = tokens
        self.term_frequencies[item_id] = Counter(tokens)
        self.total_length += len(tokens)
        for term in set(tokens):
            self.document_frequency[term] += 1

    def remove(self, item_id: Any) -> bool:
        tokens = self.documents.pop(item_id, None)
        if tokens is None:
            return False
        self.term_frequencies.pop(item_id, None)
        self.total_length -= len(tokens)
        for term in set(tokens):
            self.document_frequency[term] -= 1
            if self.document_frequency[term] <= 0:
                del self.document_frequency[term]
        return True

    def clear(self):
        self.documents.clear()
        self.term_frequencies.clear()
        self.document_frequency.clear()
        self.total_length = 0

    def search(self, query: str, k: int | None = 5) -> list[tuple[float, Any]]:
        query_tokens = self.tokenize(query)
        if not query_tokens or not self.documents:
            return []

        n_docs = len(self.documents)
        avgdl = self.total_length / n_docs if n_docs else 0.0
        query_terms = set(query_tokens)
        scores: dict[Any, float] = defaultdict(float)

        for term in query_terms:
            df = self.document_frequency.get(term, 0)
            if df == 0:
                continue
            # BM25+ style idf avoids negative values for very common terms.
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for item_id, frequencies in self.term_frequencies.items():
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                dl = len(self.documents[item_id])
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / avgdl if avgdl else 0.0)
                )
                scores[item_id] += idf * (
                    (tf * (self.k1 + 1.0)) / denominator
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
        if k is None:
            return [(score, item_id) for item_id, score in ranked]
        return [(score, item_id) for item_id, score in ranked[: max(k, 0)]]
