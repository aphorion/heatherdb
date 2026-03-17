"""Random vector vocabulary — every word gets a unique high-dim name tag."""

from __future__ import annotations

import json
import os
import re

import numpy as np


class Vocabulary:
    """Maps words to fixed random vectors. Deterministic per word via hashing."""

    def __init__(self, dimension: int = 384, cache_path: str = "vocab.json"):
        self.dimension = dimension
        self.cache_path = cache_path
        self._vectors: dict[str, list[float]] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self._vectors = json.load(f)

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self._vectors, f)

    def _make_vector(self, word: str) -> list[float]:
        """Generate a deterministic random unit vector for a word."""
        import hashlib
        h = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float64)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def get(self, word: str) -> list[float]:
        """Get the vector for a word. Creates one if it doesn't exist."""
        word = word.lower().strip()
        if word not in self._vectors:
            self._vectors[word] = self._make_vector(word)
        return self._vectors[word]

    def nearest(self, vec: list[float], top_k: int = 10,
                exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """Find the nearest words to a vector by cosine similarity."""
        v = np.array(vec, dtype=np.float64)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return []

        results = []
        for word, wvec in self._vectors.items():
            if exclude and word in exclude:
                continue
            w = np.array(wvec, dtype=np.float64)
            sim = float(np.dot(v, w) / (norm_v * np.linalg.norm(w)))
            results.append((word, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self):
        self._save_cache()

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def words(self) -> list[str]:
        return list(self._vectors.keys())


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. Lowercased."""
    text = text.lower()
    # Split on non-alphanumeric, keep words only
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


# Common stop words to skip in context windows
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "up", "about", "also",
}
