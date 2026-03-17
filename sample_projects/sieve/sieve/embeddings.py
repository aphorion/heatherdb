"""Sentence-transformer embeddings with SQLite record cache."""

import sqlite3
import struct
import time

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    def __init__(self, db_path: str = "sieve.db"):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dim = 384
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._db.commit()

    def embed(self, text: str) -> list[float]:
        """Embed text into a 384-dim vector."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once (faster than one-by-one)."""
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def store(self, text: str, vector: list[float]):
        """Save a text-vector mapping to the SQLite cache."""
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._db.execute(
            "INSERT INTO records (text, vector_blob, created_at) VALUES (?, ?, ?)",
            (text, blob, time.time()),
        )
        self._db.commit()

    def store_batch(self, texts: list[str], vectors: list[list[float]]):
        """Save multiple text-vector mappings at once."""
        now = time.time()
        rows = []
        for text, vec in zip(texts, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            rows.append((text, blob, now))
        self._db.executemany(
            "INSERT INTO records (text, vector_blob, created_at) VALUES (?, ?, ?)",
            rows,
        )
        self._db.commit()

    def find_nearest(self, vector: list[float], k: int = 3) -> list[tuple[str, float]]:
        """Find the k nearest records by cosine similarity. Returns (text, similarity) pairs."""
        query = np.array(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        rows = self._db.execute("SELECT text, vector_blob FROM records").fetchall()
        if not rows:
            return []

        scored = []
        for text, blob in rows:
            stored = np.array(struct.unpack(f"{self._dim}f", blob), dtype=np.float32)
            stored_norm = np.linalg.norm(stored)
            if stored_norm == 0:
                continue
            sim = float(np.dot(query, stored) / (query_norm * stored_norm))
            scored.append((text, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def count(self) -> int:
        """Return number of stored records."""
        return self._db.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def close(self):
        self._db.close()
