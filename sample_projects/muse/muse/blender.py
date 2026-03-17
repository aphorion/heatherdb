"""Core Muse engine: concept blending via EAM reconstruction."""

import random
from dataclasses import dataclass

import anthropic
import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class Ingredient:
    text: str
    similarity: float

    @property
    def strength(self) -> str:
        if self.similarity > 0.8:
            return "strong"
        elif self.similarity > 0.6:
            return "moderate"
        return "faint"


@dataclass
class BlendResult:
    inputs: list[str]
    nearby: list[Ingredient]
    idea: str


SYSTEM_PROMPT = (
    "You are Muse, a creative idea generator. You receive concepts that have been "
    "blended together through an associative memory system (Elastic Associative Memory). "
    "The system doesn't just find the nearest match — it reconstructs a new pattern "
    "from the interference of everything stored.\n\n"
    "Your job: take the input concepts and the nearby associations the memory surfaced, "
    "and generate a creative idea that connects them. The idea should be specific, "
    "vivid, and surprising — not a generic mashup. Give it a catchy name.\n\n"
    "Format: Start with a bold title, then 2-3 sentences describing the idea. Keep it concise."
)


class Muse:
    def __init__(self, heather: HeatherClient, embeddings: EmbeddingStore):
        self.heather = heather
        self.embeddings = embeddings
        self.claude = anthropic.Anthropic()

    def add_concept(self, text: str):
        """Store a concept in HeatherDB and the local cache."""
        vec = self.embeddings.embed(text)
        self.heather.write([vec])
        self.embeddings.store(text, vec)

    def blend(self, *concepts: str, noise: float = 0.05) -> BlendResult:
        """Blend 2+ concepts by averaging their vectors and reconstructing via EAM."""
        # Get vectors for each concept (look up cached or embed fresh)
        vectors = []
        for concept in concepts:
            vec = self.embeddings.find_by_text(concept)
            if vec is None:
                vec = self.embeddings.embed(concept)
            vectors.append(np.array(vec, dtype=np.float32))

        # Average the vectors
        blended = np.mean(vectors, axis=0)

        # Add small noise for variety
        if noise > 0:
            blended += noise * np.random.randn(len(blended)).astype(np.float32)

        # Renormalize
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        # Query HeatherDB — SDM reconstructs from the superposition
        reconstructed = self.heather.read(blended.tolist())

        # Find nearest cached concepts to the reconstruction
        nearby = self.embeddings.find_nearest(reconstructed, k=5)
        ingredients = [Ingredient(text=t, similarity=s) for t, s in nearby]

        # Ask Claude to generate an idea
        idea = self._generate_idea(list(concepts), ingredients)

        return BlendResult(inputs=list(concepts), nearby=ingredients, idea=idea)

    def spark(self, concept: str) -> BlendResult:
        """Free-associate from a single concept by adding heavy noise."""
        vec = self.embeddings.find_by_text(concept)
        if vec is None:
            vec = self.embeddings.embed(concept)

        # Heavy noise (30-50%) to push away from the original
        noise_level = random.uniform(0.3, 0.5)
        noisy = np.array(vec, dtype=np.float32) + noise_level * np.random.randn(len(vec)).astype(np.float32)
        norm = np.linalg.norm(noisy)
        if norm > 0:
            noisy = noisy / norm

        reconstructed = self.heather.read(noisy.tolist())
        nearby = self.embeddings.find_nearest(reconstructed, k=5)
        ingredients = [Ingredient(text=t, similarity=s) for t, s in nearby]

        idea = self._generate_idea([concept + " (sparked)"], ingredients)
        return BlendResult(inputs=[concept], nearby=ingredients, idea=idea)

    def explore(self, steps: int = 5) -> list[tuple[str, float]]:
        """Random walk through concept space. Returns chain of (concept, similarity)."""
        start = self.embeddings.random_vector()
        if start is None:
            return []

        chain = []
        seen = set()
        current = start

        for _ in range(steps):
            noise_level = random.uniform(0.15, 0.35)
            noisy = np.array(current, dtype=np.float32) + noise_level * np.random.randn(len(current)).astype(np.float32)
            norm = np.linalg.norm(noisy)
            if norm > 0:
                noisy = noisy / norm

            reconstructed = self.heather.read(noisy.tolist())
            nearest = self.embeddings.find_nearest(reconstructed, k=5)

            picked = None
            for text, sim in nearest:
                if text not in seen:
                    picked = (text, sim)
                    break

            if picked is None:
                break

            text, sim = picked
            chain.append((text, sim))
            seen.add(text)
            current = self.embeddings.embed(text)

        return chain

    def _generate_idea(self, inputs: list[str], ingredients: list[Ingredient]) -> str:
        """Use Claude to generate a creative idea from the blend."""
        prompt = f"Input concepts being blended: {', '.join(inputs)}\n\n"
        prompt += "Nearby concepts the associative memory surfaced:\n"
        for ing in ingredients:
            prompt += f"- {ing.text} (association strength: {ing.strength}, {ing.similarity:.0%})\n"
        prompt += "\nGenerate one creative idea that emerges from this blend."

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
