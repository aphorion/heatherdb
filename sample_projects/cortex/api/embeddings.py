"""Sentence embedding with SQLite cache for deterministic, fast lookups."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
RAW_DIMS = 384
TARGET_DIMS = 128
CACHE_DB = Path(__file__).parent / ".embedding_cache.db"
PCA_PATH = Path(__file__).parent / ".pca_matrix.npy"


def _random_projection_matrix(from_dims: int, to_dims: int) -> np.ndarray:
    """Generate a stable random projection matrix for dimensionality reduction.
    Uses a fixed seed so the projection is deterministic."""
    rng = np.random.RandomState(42)
    mat = rng.randn(from_dims, to_dims).astype(np.float64)
    # Orthogonalize via QR decomposition for better preservation
    q, _ = np.linalg.qr(mat)
    return q[:, :to_dims]


class Embedder:
    def __init__(self):
        self._model = SentenceTransformer(MODEL_NAME)
        self._proj = _random_projection_matrix(RAW_DIMS, TARGET_DIMS)
        self._conn = sqlite3.connect(str(CACHE_DB), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, vec TEXT)"
        )
        self._conn.commit()

    def _reduce(self, vec_384: np.ndarray) -> list[float]:
        """Project 384d → 128d and normalize."""
        v = vec_384 @ self._proj
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        row = self._conn.execute("SELECT vec FROM cache WHERE hash=?", (h,)).fetchone()
        if row:
            return json.loads(row[0])
        raw = self._model.encode(text, normalize_embeddings=True)
        vec = self._reduce(raw)
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
            raws = self._model.encode(
                uncached_texts, normalize_embeddings=True
            )
            for idx, text, raw in zip(uncached_indices, uncached_texts, raws):
                h = hashlib.sha256(text.encode()).hexdigest()
                vec = self._reduce(raw)
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (hash, vec) VALUES (?, ?)",
                    (h, json.dumps(vec)),
                )
                results[idx] = vec
            self._conn.commit()

        return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))
