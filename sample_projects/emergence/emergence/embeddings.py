"""Sentence embedding with SQLite cache."""

import hashlib
import json
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, cache_path: str = ".embedding_cache.db"):
        self._model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims
        self._conn = sqlite3.connect(cache_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, vec TEXT)"
        )
        self._conn.commit()

    @property
    def dim(self) -> int:
        return 384

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT vec FROM cache WHERE hash=?", (h,)
        ).fetchone()
        if row:
            return json.loads(row[0])

        raw = self._model.encode(text, normalize_embeddings=True)
        vec = raw.astype(np.float64).tolist()

        self._conn.execute(
            "INSERT OR REPLACE INTO cache (hash, vec) VALUES (?, ?)",
            (h, json.dumps(vec)),
        )
        self._conn.commit()
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            h = hashlib.sha256(text.encode()).hexdigest()
            row = self._conn.execute(
                "SELECT vec FROM cache WHERE hash=?", (h,)
            ).fetchone()
            if row:
                results.append(json.loads(row[0]))
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            raw = self._model.encode(uncached_texts, normalize_embeddings=True)
            for j, idx in enumerate(uncached_indices):
                vec = raw[j].astype(np.float64).tolist()
                results[idx] = vec
                h = hashlib.sha256(uncached_texts[j].encode()).hexdigest()
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (hash, vec) VALUES (?, ?)",
                    (h, json.dumps(vec)),
                )
            self._conn.commit()

        return results
