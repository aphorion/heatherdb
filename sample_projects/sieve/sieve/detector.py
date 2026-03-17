"""Core near-duplicate detection engine using EAM reconstruction fidelity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class Match:
    text: str
    similarity: float


@dataclass
class DuplicateResult:
    text: str
    is_duplicate: bool
    is_similar: bool
    fidelity: float
    matches: list[Match] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.is_duplicate:
            return "DUPLICATE"
        elif self.is_similar:
            return "SIMILAR"
        return "UNIQUE"


@dataclass
class ScanReport:
    total: int
    duplicates: int
    similar: int
    unique: int
    results: list[DuplicateResult]


class Sieve:
    def __init__(
        self,
        heather: HeatherClient,
        embeddings: EmbeddingStore,
        duplicate_threshold: float = 0.75,
        similar_threshold: float = 0.50,
    ):
        self.heather = heather
        self.embeddings = embeddings
        self.duplicate_threshold = duplicate_threshold
        self.similar_threshold = similar_threshold

    def ingest(self, texts: list[str]) -> int:
        """Embed and store records in HeatherDB + cache. Returns count ingested."""
        if not texts:
            return 0

        # Batch embed
        vectors = self.embeddings.embed_batch(texts)

        # Write to HeatherDB
        self.heather.write(vectors)

        # Store in SQLite cache
        self.embeddings.store_batch(texts, vectors)

        return len(texts)

    def check(self, text: str, k: int = 3) -> DuplicateResult:
        """Check a single record for near-duplicates."""
        # Embed the input
        vec = self.embeddings.embed(text)
        input_vec = np.array(vec, dtype=np.float32)

        # Query HeatherDB — SDM reconstructs from stored patterns
        reconstructed = self.heather.read(vec)
        recon_vec = np.array(reconstructed, dtype=np.float32)

        # Fidelity = cosine similarity between input and reconstruction
        norm_in = np.linalg.norm(input_vec)
        norm_re = np.linalg.norm(recon_vec)
        if norm_in > 0 and norm_re > 0:
            fidelity = float(np.dot(input_vec, recon_vec) / (norm_in * norm_re))
        else:
            fidelity = 0.0

        # Find nearest cached records to the reconstruction
        nearest = self.embeddings.find_nearest(reconstructed, k=k)
        matches = [Match(text=t, similarity=s) for t, s in nearest]

        return DuplicateResult(
            text=text,
            is_duplicate=fidelity > self.duplicate_threshold,
            is_similar=fidelity > self.similar_threshold,
            fidelity=fidelity,
            matches=matches,
        )

    def bulk_check(self, texts: list[str], k: int = 3) -> list[DuplicateResult]:
        """Check multiple records for near-duplicates."""
        return [self.check(text, k=k) for text in texts]

    def scan(self, texts: list[str], k: int = 3) -> ScanReport:
        """Ingest-and-check: find duplicates within a batch.

        For each record, checks against previously ingested records,
        then ingests it. This finds duplicates within the batch itself.
        """
        results = []
        duplicates = 0
        similar = 0
        unique = 0

        for text in texts:
            # Check against what's already been stored
            if self.embeddings.count() > 0:
                result = self.check(text, k=k)
            else:
                result = DuplicateResult(
                    text=text,
                    is_duplicate=False,
                    is_similar=False,
                    fidelity=0.0,
                    matches=[],
                )

            results.append(result)

            if result.is_duplicate:
                duplicates += 1
            elif result.is_similar:
                similar += 1
            else:
                unique += 1

            # Ingest this record so subsequent checks can find it
            self.ingest([text])

        return ScanReport(
            total=len(texts),
            duplicates=duplicates,
            similar=similar,
            unique=unique,
            results=results,
        )
