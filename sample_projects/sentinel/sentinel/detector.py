"""Core anomaly detection engine using EAM reconstruction diff."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class Anomaly:
    text: str
    fidelity: float
    level: str                        # NORMAL, SUSPICIOUS, ANOMALY
    expected: str                     # nearest text to reconstruction (what SDM thinks is normal)
    expected_similarity: float        # how close reconstruction is to that text
    nearest_input: str                # nearest stored text to the INPUT (what it looks most like)
    nearest_input_similarity: float

    @property
    def deviation_summary(self) -> str:
        """Short description of what's different."""
        if self.level == "NORMAL":
            return "matches expected pattern"
        return f"expected something like: {self.expected[:80]}"


@dataclass
class TrainReport:
    count: int
    baseline_fidelity: float  # avg fidelity of training data against itself


@dataclass
class MonitorReport:
    total: int
    normal: int
    suspicious: int
    anomalies: int
    results: list[Anomaly]


class Sentinel:
    def __init__(
        self,
        heather: HeatherClient,
        embeddings: EmbeddingStore,
        anomaly_threshold: float = 0.65,
        suspicious_threshold: float = 0.80,
    ):
        self.heather = heather
        self.embeddings = embeddings
        self.anomaly_threshold = anomaly_threshold
        self.suspicious_threshold = suspicious_threshold

    def train(self, texts: list[str]) -> TrainReport:
        """Learn normal patterns. Embeds and writes to HeatherDB + cache."""
        if not texts:
            return TrainReport(count=0, baseline_fidelity=0.0)

        vectors = self.embeddings.embed_batch(texts)
        self.heather.write(vectors)
        self.embeddings.store_batch(texts, vectors)

        # Measure baseline: how well does SDM reconstruct training data?
        fidelities = []
        for vec in vectors[:min(20, len(vectors))]:  # sample up to 20
            recon = self.heather.read(vec)
            fid = self._cosine(vec, recon)
            fidelities.append(fid)

        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0

        return TrainReport(count=len(texts), baseline_fidelity=avg_fidelity)

    def check(self, text: str) -> Anomaly:
        """Check a single log/event. Returns anomaly analysis with reconstruction diff."""
        vec = self.embeddings.embed(text)
        input_vec = np.array(vec, dtype=np.float32)

        # Query SDM — what does it think this should look like?
        reconstructed = self.heather.read(vec)
        recon_vec = np.array(reconstructed, dtype=np.float32)

        # Fidelity: how well does SDM reconstruct this input?
        fidelity = self._cosine(vec, reconstructed)

        # What text does the RECONSTRUCTION correspond to?
        # This is what SDM thinks "normal" looks like for this input
        recon_nearest = self.embeddings.find_nearest(reconstructed, k=1)
        if recon_nearest:
            expected_text, expected_sim = recon_nearest[0]
        else:
            expected_text, expected_sim = "(no baseline)", 0.0

        # What stored text is the INPUT itself closest to?
        input_nearest = self.embeddings.find_nearest(vec, k=1)
        if input_nearest:
            nearest_text, nearest_sim = input_nearest[0]
        else:
            nearest_text, nearest_sim = "(no baseline)", 0.0

        # Classify
        if fidelity >= self.suspicious_threshold:
            level = "NORMAL"
        elif fidelity >= self.anomaly_threshold:
            level = "SUSPICIOUS"
        else:
            level = "ANOMALY"

        return Anomaly(
            text=text,
            fidelity=fidelity,
            level=level,
            expected=expected_text,
            expected_similarity=expected_sim,
            nearest_input=nearest_text,
            nearest_input_similarity=nearest_sim,
        )

    def monitor(self, texts: list[str]) -> MonitorReport:
        """Check a batch of events. Returns full report."""
        results = []
        normal = suspicious = anomalies = 0

        for text in texts:
            result = self.check(text)
            results.append(result)
            if result.level == "NORMAL":
                normal += 1
            elif result.level == "SUSPICIOUS":
                suspicious += 1
            else:
                anomalies += 1

        return MonitorReport(
            total=len(texts),
            normal=normal,
            suspicious=suspicious,
            anomalies=anomalies,
            results=results,
        )

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
