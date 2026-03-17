"""Sentence-transformer embeddings with SQLite thought cache."""

import sqlite3
import struct
import time

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    def __init__(self, db_path: str = "swarm.db"):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dim = 384
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                text TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._db.commit()

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def store(self, agent: str, text: str, vector: list[float]):
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._db.execute(
            "INSERT INTO thoughts (agent, text, vector_blob, created_at) VALUES (?, ?, ?, ?)",
            (agent, text, blob, time.time()),
        )
        self._db.commit()

    def store_batch(self, agent: str, texts: list[str], vectors: list[list[float]]):
        now = time.time()
        rows = []
        for text, vec in zip(texts, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            rows.append((agent, text, blob, now))
        self._db.executemany(
            "INSERT INTO thoughts (agent, text, vector_blob, created_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._db.commit()

    def find_nearest(self, vector: list[float], k: int = 5) -> list[tuple[str, str, float]]:
        """Returns (agent, text, similarity) triples."""
        query = np.array(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        rows = self._db.execute("SELECT agent, text, vector_blob FROM thoughts").fetchall()
        if not rows:
            return []

        scored = []
        for agent, text, blob in rows:
            stored = np.array(struct.unpack(f"{self._dim}f", blob), dtype=np.float32)
            stored_norm = np.linalg.norm(stored)
            if stored_norm == 0:
                continue
            sim = float(np.dot(query, stored) / (query_norm * stored_norm))
            scored.append((agent, text, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM thoughts").fetchone()[0]

    def count_by_agent(self) -> dict[str, int]:
        rows = self._db.execute(
            "SELECT agent, COUNT(*) FROM thoughts GROUP BY agent"
        ).fetchall()
        return {agent: count for agent, count in rows}

    def close(self):
        self._db.close()
