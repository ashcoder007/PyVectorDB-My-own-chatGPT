import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


class SQLiteStorage:
    """Persistent storage for vectors and JSON metadata."""

    def __init__(self, path: str | Path, collection: str = "default"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not collection or not isinstance(collection, str):
            raise ValueError("collection must be a non-empty string.")
        self.collection = collection
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT NOT NULL,
                item_id TEXT NOT NULL,
                id_type TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY (collection, item_id),
                FOREIGN KEY (collection) REFERENCES collections(name)
                    ON DELETE CASCADE
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_collection "
            "ON vectors(collection)"
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO collections(name) VALUES (?)",
            (self.collection,),
        )
        self.connection.commit()

    @staticmethod
    def _encode_id(item_id: int | str) -> tuple[str, str]:
        if isinstance(item_id, bool):
            return str(int(item_id)), "int"
        if isinstance(item_id, int):
            return str(item_id), "int"
        if isinstance(item_id, str):
            return item_id, "str"
        raise TypeError("item_id must be an int or str.")

    @staticmethod
    def _decode_id(value: str, id_type: str) -> int | str:
        return int(value) if id_type == "int" else value

    def upsert(self, item_id, vector: np.ndarray, metadata: dict[str, Any]):
        key, id_type = self._encode_id(item_id)
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Vector must be one-dimensional.")
        metadata_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO vectors(collection, item_id, id_type, dimension, vector, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection, item_id) DO UPDATE SET
                id_type=excluded.id_type,
                dimension=excluded.dimension,
                vector=excluded.vector,
                metadata=excluded.metadata
            """,
            (
                self.collection,
                key,
                id_type,
                int(array.size),
                sqlite3.Binary(array.tobytes()),
                metadata_json,
            ),
        )
        self.connection.commit()

    def delete(self, item_id) -> bool:
        key, _ = self._encode_id(item_id)
        cursor = self.connection.execute(
            "DELETE FROM vectors WHERE collection=? AND item_id=?",
            (self.collection, key),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def get(self, item_id):
        key, _ = self._encode_id(item_id)
        return self.connection.execute(
            """
            SELECT item_id, id_type, dimension, vector, metadata
            FROM vectors WHERE collection=? AND item_id=?
            """,
            (self.collection, key),
        ).fetchone()

    def load_all(self):
        rows = self.connection.execute(
            """
            SELECT item_id, id_type, dimension, vector, metadata
            FROM vectors WHERE collection=? ORDER BY rowid
            """,
            (self.collection,),
        ).fetchall()

        result = []
        for item_id, id_type, dimension, vector_blob, metadata_json in rows:
            vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
            if vector.size != dimension:
                raise ValueError(
                    f"Corrupt vector data for item {item_id}: "
                    f"expected {dimension} values, found {vector.size}."
                )
            metadata = json.loads(metadata_json)
            result.append(
                (
                    self._decode_id(item_id, id_type),
                    vector,
                    metadata,
                )
            )
        return result

    def collections(self):
        rows = self.connection.execute(
            "SELECT name FROM collections ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]

    def count(self):
        return self.connection.execute(
            "SELECT COUNT(*) FROM vectors WHERE collection=?",
            (self.collection,),
        ).fetchone()[0]

    def clear(self):
        self.connection.execute(
            "DELETE FROM vectors WHERE collection=?",
            (self.collection,),
        )
        self.connection.commit()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
