"""Cipher engine — code pattern completion through SDM.

How it works:
  1. Code snippets are stored with their descriptions
  2. Each snippet is embedded (description + code combined) and written to SDM
  3. Query with a natural language description → SDM reconstructs a pattern
  4. The reconstruction is a BLEND of all similar snippets — not any single one
  5. Find nearest cached snippets to the reconstruction → those are the building blocks
  6. The reconstruction itself represents the "typical" code for this task

Key difference from vector DB (Copilot-style retrieval):
  - Vector DB: "here's the single closest snippet I have"
  - SDM: "here's what code for this kind of task TYPICALLY looks like,
          blended from all similar patterns I've seen"

The reconstruction captures PATTERNS, not specific implementations.
Frequently used patterns form deeper basins → they dominate reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class CodeResult:
    """Result of a code pattern query."""
    query: str
    fidelity: float            # how strongly SDM recognizes this pattern
    nearest: list[tuple[str, str, float]]  # (description, code, similarity)
    blend_nearest: list[tuple[str, str, float]]  # nearest to reconstruction


@dataclass
class BlendResult:
    """Result of blending multiple concepts."""
    concepts: list[str]
    fidelity: float
    nearest: list[tuple[str, str, float]]


class Cipher:
    """Code pattern completion engine."""

    def __init__(self, heather: HeatherClient, embeddings: EmbeddingStore):
        self.heather = heather
        self.embeddings = embeddings
        self.snippets_stored = 0

    def ingest(self, description: str, code: str, language: str = "python"):
        """Store a code snippet. Embeds description+code together."""
        # Embed the combined text for richer representation
        combined = f"{description}\n\n{code}"
        vec = self.embeddings.embed(combined)
        self.heather.write([vec])
        self.embeddings.store(description, code, vec, language)
        self.snippets_stored += 1

    def ingest_batch(self, snippets: list[dict]):
        """Batch ingest: [{"description": ..., "code": ..., "language": ...}, ...]."""
        texts = [f"{s['description']}\n\n{s['code']}" for s in snippets]
        vecs = self.embeddings.embed_batch(texts)

        self.heather.write(vecs)

        items = []
        for s, v in zip(snippets, vecs):
            items.append((s["description"], s["code"], v,
                          s.get("language", "python")))
        self.embeddings.store_batch(items)
        self.snippets_stored += len(snippets)

    def query(self, description: str, k: int = 5) -> CodeResult:
        """Query with a natural language description.

        Returns both:
        - Direct nearest matches (what a vector DB would give you)
        - Reconstruction nearest matches (what EAM's pattern completion gives)
        """
        vec = self.embeddings.embed(description)

        # Direct nearest (vector DB equivalent)
        direct_nearest = self.embeddings.find_nearest(vec, k=k)

        # EAM reconstruction
        reconstructed = self.heather.read(vec)

        # Fidelity
        v = np.array(vec, dtype=np.float64)
        r = np.array(reconstructed, dtype=np.float64)
        fidelity = float(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r)))

        # Find nearest to the RECONSTRUCTION
        blend_nearest = self.embeddings.find_nearest(reconstructed, k=k)

        return CodeResult(
            query=description,
            fidelity=fidelity,
            nearest=direct_nearest,
            blend_nearest=blend_nearest,
        )

    def blend(self, *descriptions: str, k: int = 5) -> BlendResult:
        """Blend multiple concepts — find code at their intersection.

        Averages the EAM reconstructions of each concept, then finds
        nearest snippets. The result represents code that lives at
        the intersection of all input concepts.
        """
        reconstructions = []
        for desc in descriptions:
            vec = self.embeddings.embed(desc)
            rec = self.heather.read(vec)
            reconstructions.append(np.array(rec, dtype=np.float64))

        blended = sum(reconstructions) / len(reconstructions)
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        # Fidelity of the blend
        fidelity_scores = []
        for rec in reconstructions:
            f = float(np.dot(blended, rec) / (np.linalg.norm(blended) * np.linalg.norm(rec)))
            fidelity_scores.append(f)

        nearest = self.embeddings.find_nearest(blended.tolist(), k=k)

        return BlendResult(
            concepts=list(descriptions),
            fidelity=float(np.mean(fidelity_scores)),
            nearest=nearest,
        )

    def complete(self, partial_code: str, description: str = "",
                 k: int = 5) -> CodeResult:
        """Given partial code (+ optional description), complete the pattern.

        The EAM reconstruction of partial code reveals what typically
        comes with this kind of code — the "missing pieces."
        """
        combined = f"{description}\n\n{partial_code}" if description else partial_code
        return self.query(combined, k=k)
