"""SDM-based sensorimotor intuition: danger memory.

Biological model:
  1. Robot moves forward by default (exploration drive)
  2. When it crashes, the sensor pattern is stored as a DANGER memory
  3. Next time similar sensors fire, SDM recognizes the pattern — "danger!"
  4. High fidelity = familiar danger → turn toward open space
  5. Low fidelity = unfamiliar, probably safe → keep going

One SDM query per step. No action encoding. The EAM is an intuition engine
that learns what danger FEELS like.
"""

from __future__ import annotations

import numpy as np

from .client import HeatherClient

# 16 features per sensor × 8 sensors = 128 dims
DIMENSION = 128


def encode_sensor(distances: list[float]) -> list[float]:
    """Encode 8 sensor distances into 128 dims.
    Rich feature expansion so EAM can discriminate subtle differences."""
    parts = []
    for d in distances:
        inv = 1.0 - d
        parts.extend([
            d,              # raw distance
            inv,            # inverse (closeness)
            d * d,          # squared
            inv * inv,      # inverse squared
            d * d * d,      # cubed
            inv * inv * inv,
            d * inv,        # interaction term
            abs(d - 0.5),   # deviation from mid
            min(d, inv),    # proximity to nearest extreme
            max(d, inv),    # distance from nearest extreme
            1.0 if d < 0.15 else 0.0,   # wall detector
            1.0 if d < 0.30 else 0.0,   # near-wall detector
            1.0 if d > 0.50 else 0.0,   # open-space detector
            1.0 if d > 0.75 else 0.0,   # wide-open detector
            d ** 0.5 if d > 0 else 0.0,  # sqrt
            inv ** 0.5 if inv > 0 else 0.0,  # sqrt inverse
        ])
    vec = np.array(parts, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


class Brain:
    """SDM-based danger intuition.

    The EAM stores danger patterns — sensor states that preceded crashes.
    At decision time, one query tells the robot: is this familiar danger?
    """

    def __init__(self, heather: HeatherClient):
        self.heather = heather
        self.memories = 0
        self.danger_threshold = 0.75  # above this = recognized danger
        self.recent_positions: list[tuple[int, int]] = []
        self.boredom_window = 20  # how far back to check
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.frustration = 0.0

    def remember_danger(self, distances: list[float]):
        """Store a danger pattern — this sensor state led to a crash."""
        vec = encode_sensor(distances)
        self.heather.write([vec])
        self.memories += 1

    def remember_dangers(self, patterns: list[list[float]]):
        """Batch store danger patterns."""
        vectors = [encode_sensor(d) for d in patterns]
        if vectors:
            self.heather.write(vectors)
            self.memories += len(vectors)

    def sense_danger(self, distances: list[float]) -> float:
        """Query SDM: how dangerous does this feel?
        Returns fidelity score — how strongly the EAM recognizes this
        sensor pattern as a known danger."""
        vec = encode_sensor(distances)
        reconstructed = self.heather.read(vec)
        return cosine_similarity(vec, reconstructed)

    def update_position(self, x: int, y: int):
        """Track position. Revisits build frustration, new cells reduce it."""
        pos = (x, y)
        self.recent_positions.append(pos)
        if len(self.recent_positions) > self.boredom_window:
            self.recent_positions = self.recent_positions[-self.boredom_window:]

        visits = self.visit_counts.get(pos, 0) + 1
        self.visit_counts[pos] = visits

        if visits > 1:
            # Been here before — frustration builds
            self.frustration = min(1.0, self.frustration + 0.15)
        else:
            # New cell! Curiosity satisfied, frustration drops
            self.frustration = max(0.0, self.frustration - 0.3)

    def decide(self, distances: list[float]) -> tuple[str, float]:
        """Decide what to do based on danger intuition and boredom.
        Returns (action, danger_level).

        - Bored (stuck in loop) → force a new direction
        - High danger → turn toward open space
        - Low danger → go forward (exploration drive)
        """
        danger = 0.0

        if self.memories > 5:
            danger = self.sense_danger(distances)

        # Frustration override — been here too many times, try something new
        if self.frustration > 0.6:
            import random
            self.frustration = 0.0  # reset after breaking out
            # Pick a random open direction to escape the rut
            options = []
            if distances[0] > 0.2:
                options.append("forward")
            if distances[6] > 0.2:
                options.append("turn_left")
            if distances[2] > 0.2:
                options.append("turn_right")
            if not options:
                options = ["turn_left", "turn_right"]
            return random.choice(options), danger

        if danger > self.danger_threshold:
            # Danger recognized — turn toward open space
            left = distances[6]   # left sensor
            right = distances[2]  # right sensor
            if left > right:
                return "turn_left", danger
            elif right > left:
                return "turn_right", danger
            else:
                return "turn_left", danger  # arbitrary tiebreak

        # Safe — go forward (exploration drive)
        return "forward", danger
