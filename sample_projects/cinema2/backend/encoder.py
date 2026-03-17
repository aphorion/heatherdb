"""Movie metadata → 128d vector encoding using sentence embeddings + random projection."""

import hashlib

import numpy as np

DIMENSION = 128
_SRC_DIM = 384

_model = None
_projection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_projection() -> np.ndarray:
    """Fixed random projection matrix 384→128, QR orthogonalized."""
    global _projection
    if _projection is None:
        rng = np.random.RandomState(20240101)
        raw = rng.randn(_SRC_DIM, DIMENSION)
        q, _ = np.linalg.qr(raw)
        _projection = q
    return _projection


def _project(vec_384: np.ndarray) -> np.ndarray:
    """Project a 384d vector to 128d."""
    return vec_384 @ _get_projection()


def feature_vector(feature_type: str, value: str) -> np.ndarray:
    """Generate a deterministic random unit vector for a typed feature (128d)."""
    key = f"{feature_type}:{value.lower().strip()}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIMENSION).astype(np.float64)
    return vec / np.linalg.norm(vec)


def embed_plot(plot: str) -> np.ndarray:
    """Embed a plot string → project to 128d."""
    if not plot or plot == "N/A":
        return np.zeros(DIMENSION, dtype=np.float64)
    model = _get_model()
    vec_384 = model.encode(plot, normalize_embeddings=True).astype(np.float64)
    projected = _project(vec_384)
    n = np.linalg.norm(projected)
    return projected / n if n > 1e-10 else projected


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-10:
        return vec
    return vec / n


def encode_movie(movie: dict) -> list[float]:
    """Encode a movie dict into a 128d vector."""
    components = np.zeros(DIMENSION, dtype=np.float64)

    plot = movie.get("plot", "")
    plot_vec = embed_plot(plot)
    if np.linalg.norm(plot_vec) > 1e-10:
        components += 0.45 * normalize(plot_vec)

    genres = movie.get("genres", [])
    if genres:
        genre_sum = sum(feature_vector("genre", g) for g in genres)
        components += 0.25 * normalize(genre_sum)

    director = movie.get("director", "")
    if director and director != "N/A":
        components += 0.15 * feature_vector("director", director)

    actors = movie.get("actors", [])
    if actors:
        actor_sum = sum(feature_vector("actor", a) for a in actors[:4])
        components += 0.15 * normalize(actor_sum)

    return normalize(components).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def genre_affinities(fingerprint: list[float], genres: list[str]) -> dict[str, float]:
    result = {}
    for g in genres:
        gvec = feature_vector("genre", g)
        result[g] = cosine_similarity(fingerprint, gvec.tolist())
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
