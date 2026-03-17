"""Core Whisper engine: compress documents into EAM, reconstruct from any angle."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class Fragment:
    """A sentence fragment nearest to the reconstruction."""
    text: str
    source: str
    similarity: float


@dataclass
class Reconstruction:
    """What SDM reconstructed for a query."""
    query: str
    fidelity: float           # how strongly SDM responded
    fragments: list[Fragment]  # nearest stored sentences to the reconstruction
    interpretation: str        # Claude's interpretation of the reconstruction


INTERPRET_PROMPT = (
    "You are interpreting a lossy reconstruction from an associative memory. "
    "A document was compressed into a Elastic Associative Memory — all sentences "
    "superimposed into shared storage. When queried, the memory reconstructs a "
    "pattern influenced by ALL stored sentences, not just the closest match.\n\n"
    "You're given the query and the nearest stored sentences to the reconstruction. "
    "These fragments represent what the memory 'remembers' about this topic — "
    "frequent themes are vivid, minor details are faded.\n\n"
    "Synthesize these fragments into a coherent answer to the query. "
    "Speak as if recalling from memory — confident about vivid parts, "
    "approximate about faded ones. Be concise (2-4 sentences)."
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based splitter."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short fragments
        if len(s) > 20:
            sentences.append(s)
    return sentences


def load_text_file(path: str) -> str:
    """Load a text file."""
    return Path(path).read_text(encoding="utf-8")


class Whisper:
    def __init__(self, heather: HeatherClient, embeddings: EmbeddingStore):
        self.heather = heather
        self.embeddings = embeddings
        self.claude = anthropic.Anthropic()

    def absorb(self, text: str, source: str = "document") -> int:
        """Compress a document into EAM. Returns number of sentences absorbed."""
        sentences = split_sentences(text)
        if not sentences:
            return 0

        # Embed all sentences
        vectors = self.embeddings.embed_batch(sentences)

        # Write ALL to SDM — they superimpose into shared memory
        self.heather.write(vectors)

        # Cache text-vector mappings for later lookup
        self.embeddings.store_batch(source, sentences, vectors)

        return len(sentences)

    def absorb_file(self, filepath: str) -> int:
        """Load and absorb a text file."""
        text = load_text_file(filepath)
        source = Path(filepath).name
        return self.absorb(text, source=source)

    def recall(self, query: str, k: int = 5) -> Reconstruction:
        """Query the compressed memory. Returns reconstructed understanding."""
        query_vec = self.embeddings.embed(query)

        # Query SDM — reconstruction is influenced by ALL absorbed sentences
        reconstructed = self.heather.read(query_vec)

        # Fidelity: how strongly did the memory respond?
        fidelity = self._cosine(query_vec, reconstructed)

        # Find nearest stored sentences to the RECONSTRUCTION
        # (not to the query — this is the key difference from vector search)
        nearest = self.embeddings.find_nearest(reconstructed, k=k)
        fragments = [
            Fragment(text=text, source=src, similarity=sim)
            for src, text, sim in nearest
        ]

        # Also probe with noise to explore the reconstruction's neighborhood
        noisy_fragments = self._probe_neighborhood(query_vec, k=3)
        seen = {f.text for f in fragments}
        for f in noisy_fragments:
            if f.text not in seen:
                fragments.append(f)
                seen.add(f.text)

        # Interpret the reconstruction
        interpretation = self._interpret(query, fragments)

        return Reconstruction(
            query=query,
            fidelity=fidelity,
            fragments=fragments,
            interpretation=interpretation,
        )

    def _probe_neighborhood(self, query_vec: list[float], k: int = 3) -> list[Fragment]:
        """Add noise and reconstruct to explore nearby attractors."""
        import random
        noise_level = random.uniform(0.1, 0.25)
        noisy = np.array(query_vec, dtype=np.float32)
        noisy += noise_level * np.random.randn(len(noisy)).astype(np.float32)
        norm = np.linalg.norm(noisy)
        if norm > 0:
            noisy = noisy / norm

        reconstructed = self.heather.read(noisy.tolist())
        nearest = self.embeddings.find_nearest(reconstructed, k=k)
        return [
            Fragment(text=text, source=src, similarity=sim)
            for src, text, sim in nearest
        ]

    def _interpret(self, query: str, fragments: list[Fragment]) -> str:
        """Use Claude to interpret the reconstruction."""
        prompt = f"Query: {query}\n\n"
        prompt += "Reconstructed fragments (ordered by vividness):\n"
        for f in fragments:
            label = "vivid" if f.similarity > 0.8 else "clear" if f.similarity > 0.6 else "faded"
            prompt += f"  [{label} {f.similarity:.0%}] {f.text}\n"
        prompt += "\nWhat does the memory recall about this query?"

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            system=INTERPRET_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
