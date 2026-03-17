"""Embedding store — sentence-transformers + SQLite cache for code snippets."""

from __future__ import annotations

import os
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """Embeds text/code and caches in SQLite for reverse lookup."""

    def __init__(self, db_path: str = "cipher.db",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                language TEXT,
                vector BLOB NOT NULL
            )
        """)
        self.conn.commit()

    def embed(self, text: str) -> list[float]:
        """Embed a text string."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def store(self, description: str, code: str, vector: list[float],
              language: str = "python"):
        """Store a snippet with its embedding."""
        blob = np.array(vector, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO snippets (description, code, language, vector) "
            "VALUES (?, ?, ?, ?)",
            (description, code, language, blob),
        )
        self.conn.commit()

    def store_batch(self, items: list[tuple[str, str, list[float], str]]):
        """Batch store: [(description, code, vector, language), ...]."""
        rows = []
        for desc, code, vec, lang in items:
            blob = np.array(vec, dtype=np.float32).tobytes()
            rows.append((desc, code, lang, blob))
        self.conn.executemany(
            "INSERT INTO snippets (description, code, language, vector) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def find_nearest(self, vec: list[float], k: int = 5) -> list[tuple[str, str, float]]:
        """Find nearest snippets by cosine similarity.
        Returns [(description, code, similarity), ...]."""
        query = np.array(vec, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        rows = self.conn.execute(
            "SELECT description, code, vector FROM snippets"
        ).fetchall()

        results = []
        for desc, code, blob in rows:
            stored = np.frombuffer(blob, dtype=np.float32)
            sim = float(np.dot(query, stored) / (query_norm * np.linalg.norm(stored)))
            results.append((desc, code, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]

    def close(self):
        self.conn.close()
