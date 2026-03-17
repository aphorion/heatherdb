"""Oracle diagnostic engine — pattern completion through SDM.

The core idea: store many patient symptom profiles in the EAM.
When given a PARTIAL profile (some symptoms observed, others unknown),
the EAM reconstructs the FULL pattern — predicting what other symptoms
are likely present based on the statistical structure of all stored profiles.

This is pattern completion, not lookup. The EAM doesn't find the "nearest
patient" — it reconstructs the composite pattern from the interference
of ALL similar profiles.
"""

from __future__ import annotations

import numpy as np

from .client import HeatherClient
from .symptoms import (
    SYMPTOMS, DIMENSION, INDEX_TO_SYMPTOM, CATEGORIES,
    encode_profile, decode_profile,
)


class Diagnosis:
    """Result of a diagnostic query."""

    def __init__(self, observed: dict[str, float], predicted: dict[str, float],
                 fidelity: float):
        self.observed = observed       # symptoms the user reported
        self.predicted = predicted     # symptoms SDM predicts (the completion)
        self.fidelity = fidelity       # how strongly SDM recognized the pattern

    @property
    def new_predictions(self) -> dict[str, float]:
        """Symptoms predicted by SDM that weren't in the original query."""
        return {k: v for k, v in self.predicted.items()
                if k not in self.observed}

    @property
    def confirmed(self) -> dict[str, float]:
        """Symptoms that were observed AND appear in the reconstruction."""
        return {k: v for k, v in self.predicted.items()
                if k in self.observed}


class Oracle:
    """Diagnostic pattern completion engine."""

    def __init__(self, heather: HeatherClient):
        self.heather = heather
        self.profiles_stored = 0

    def store_patient(self, symptoms: dict[str, float],
                      condition: str | None = None):
        """Store a patient's symptom profile in the EAM.
        symptoms: {symptom_name: severity 0.0-1.0}
        condition: optional label (stored locally, not in SDM)"""
        vec = encode_profile(symptoms)
        self.heather.write([vec])
        self.profiles_stored += 1

    def store_patients(self, patients: list[dict[str, float]]):
        """Batch store multiple patient profiles."""
        vectors = [encode_profile(p) for p in patients]
        if vectors:
            self.heather.write(vectors)
            self.profiles_stored += len(vectors)

    def diagnose(self, observed_symptoms: dict[str, float],
                 prediction_threshold: float = 0.1) -> Diagnosis:
        """Given partial symptoms, predict the full profile.

        The EAM reconstructs what a "typical" patient with these
        symptoms looks like — including symptoms not yet observed.
        """
        query_vec = encode_profile(observed_symptoms)
        reconstructed = self.heather.read(query_vec)

        # Fidelity: how well does SDM recognize this pattern?
        q = np.array(query_vec, dtype=np.float64)
        r = np.array(reconstructed, dtype=np.float64)
        dot = np.dot(q, r)
        nq, nr = np.linalg.norm(q), np.linalg.norm(r)
        fidelity = float(dot / (nq * nr)) if nq > 0 and nr > 0 else 0.0

        # Decode the reconstruction
        predicted = decode_profile(reconstructed, threshold=prediction_threshold)

        return Diagnosis(
            observed=observed_symptoms,
            predicted=predicted,
            fidelity=fidelity,
        )

    def differential(self, observed_symptoms: dict[str, float],
                     top_k: int = 5) -> list[tuple[str, float]]:
        """Suggest what to check next — the symptoms with highest
        predicted severity that haven't been observed yet."""
        result = self.diagnose(observed_symptoms, prediction_threshold=0.05)
        new = result.new_predictions
        sorted_preds = sorted(new.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]

    def compare_conditions(self, observed_symptoms: dict[str, float],
                           condition_profiles: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
        """Compare observed symptoms against known condition profiles.
        Returns conditions sorted by similarity to the EAM reconstruction."""
        result = self.diagnose(observed_symptoms)
        rec = np.array(encode_profile(result.predicted), dtype=np.float64)

        scores = []
        for name, profile in condition_profiles.items():
            cond_vec = np.array(encode_profile(profile), dtype=np.float64)
            dot = np.dot(rec, cond_vec)
            nrec, nc = np.linalg.norm(rec), np.linalg.norm(cond_vec)
            sim = float(dot / (nrec * nc)) if nrec > 0 and nc > 0 else 0.0
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
