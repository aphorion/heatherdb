"""Movie metadata → 384d vector encoding using sentence embeddings + random projections."""

import hashlib
import re

import numpy as np

DIMENSION = 384

_model = None


def _get_model():
    """Lazy-load sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def feature_vector(feature_type: str, value: str) -> np.ndarray:
    """Generate a deterministic random unit vector for a typed feature."""
    key = f"{feature_type}:{value.lower().strip()}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIMENSION).astype(np.float64)
    return vec / np.linalg.norm(vec)


def embed_plot(plot: str) -> np.ndarray:
    """Embed a plot string using sentence-transformers → 384d vector."""
    if not plot or plot == "N/A":
        return np.zeros(DIMENSION, dtype=np.float64)
    model = _get_model()
    vec = model.encode(plot, normalize_embeddings=True)
    return vec.astype(np.float64)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize to unit vector."""
    n = np.linalg.norm(vec)
    if n < 1e-10:
        return vec
    return vec / n


def encode_movie(movie: dict) -> list[float]:
    """Encode a movie dict into a 384d vector.

    Combines semantic plot embedding (45%) with structured feature projections.
    movie dict should have: genres, plot, director, actors
    """
    components = np.zeros(DIMENSION, dtype=np.float64)

    # Plot semantic embedding (weight 0.45)
    plot = movie.get("plot", "")
    plot_vec = embed_plot(plot)
    if np.linalg.norm(plot_vec) > 1e-10:
        components += 0.45 * normalize(plot_vec)

    # Genres (weight 0.25)
    genres = movie.get("genres", [])
    if genres:
        genre_sum = sum(feature_vector("genre", g) for g in genres)
        components += 0.25 * normalize(genre_sum)

    # Director (weight 0.15)
    director = movie.get("director", "")
    if director and director != "N/A":
        components += 0.15 * feature_vector("director", director)

    # Actors (weight 0.15)
    actors = movie.get("actors", [])
    if actors:
        actor_sum = sum(feature_vector("actor", a) for a in actors[:4])
        components += 0.15 * normalize(actor_sum)

    return normalize(components).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def genre_affinities(fingerprint: list[float], genres: list[str]) -> dict[str, float]:
    """Compute cosine similarity of a fingerprint against genre vectors."""
    result = {}
    for g in genres:
        gvec = feature_vector("genre", g)
        result[g] = cosine_similarity(fingerprint, gvec.tolist())
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
