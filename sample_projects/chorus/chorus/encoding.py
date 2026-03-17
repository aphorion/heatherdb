"""Encode signal windows as 128-dim vectors for SDM.

The signal IS the vector (128 samples → 128 dims, unit normalized).

Position binding: each window is multiplied element-wise by a
position-dependent random ±1 key. This makes same-position copies
similar (noise cancels in superposition) while different positions
become nearly orthogonal (no phase confusion).

This is "binding" from holographic associative memory — binding
content to an address.
"""

from __future__ import annotations

import numpy as np

DIMENSION = 128
WINDOW_SIZE = 128


def _position_key(position: int) -> np.ndarray:
    """Deterministic random ±1 vector for a window position."""
    rng = np.random.RandomState(seed=position * 7919 + 104729)
    return rng.choice([-1.0, 1.0], size=DIMENSION)


def encode_window(window: np.ndarray, position: int = 0) -> list[float]:
    """Encode a 128-sample window as a position-bound unit vector."""
    vec = np.array(window, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    # Bind to position
    vec = vec * _position_key(position)
    return vec.tolist()


def decode_window(vec: list[float], position: int = 0,
                  target_norm: float = 1.0) -> np.ndarray:
    """Decode an EAM reconstruction back to a waveform.

    Unbinds the position key, then scales to target magnitude.
    """
    v = np.array(vec, dtype=np.float64)
    # Unbind position (multiply by same ±1 key — self-inverse)
    v = v * _position_key(position)
    # Scale back to original magnitude
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v * (target_norm / v_norm)
    return v


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))
