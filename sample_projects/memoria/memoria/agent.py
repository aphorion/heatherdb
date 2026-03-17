"""Core Memoria agent: associative memory + Claude conversation."""

import random
from dataclasses import dataclass

import anthropic
import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore


@dataclass
class Memory:
    text: str
    confidence: float  # cosine similarity between reconstruction and cached vector

    @property
    def label(self) -> str:
        if self.confidence > 0.85:
            return "vivid memory"
        elif self.confidence > 0.7:
            return "clear memory"
        elif self.confidence > 0.5:
            return "vague impression"
        return "faded"


SYSTEM_PROMPT = (
    "You are Memoria, an AI with associative memory powered by a Elastic Associative Memory. "
    "You don't have perfect recall — some memories are vivid, others are vague impressions. "
    "When you remember something clearly, speak with confidence. When a memory is vague, "
    "express appropriate uncertainty. You find connections between ideas naturally, "
    "the way human memory works — through association, not exact lookup.\n\n"
    "Keep responses conversational and concise."
)


class Memoria:
    def __init__(self, heather: HeatherClient, embeddings: EmbeddingStore):
        self.heather = heather
        self.embeddings = embeddings
        self.claude = anthropic.Anthropic()
        self.conversation: list[dict] = []

    def remember(self, text: str):
        """Encode text into HeatherDB and cache the text-vector mapping."""
        vec = self.embeddings.embed(text)
        self.heather.write([vec])
        self.embeddings.store(text, vec)

    def recall(self, query: str, k: int = 3) -> list[Memory]:
        """Query HeatherDB with text, find nearest cached texts to the reconstruction."""
        query_vec = self.embeddings.embed(query)
        reconstructed = self.heather.read(query_vec)
        nearest = self.embeddings.find_nearest(reconstructed, k=k)
        memories = []
        for text, sim in nearest:
            if sim > 0.5:
                memories.append(Memory(text=text, confidence=sim))
        return memories

    def chat(self, user_message: str) -> tuple[str, list[Memory]]:
        """Full loop: recall → build prompt → Claude → remember. Returns (response, recalled_memories)."""
        # Recall relevant memories
        memories = self.recall(user_message)

        # Build system prompt with memories
        system = SYSTEM_PROMPT
        if memories:
            system += "\n\nYour recalled memories (use these naturally in conversation):\n"
            for mem in memories:
                system += f"- [{mem.label}, confidence {mem.confidence:.0%}]: {mem.text}\n"

        # Add user message to conversation
        self.conversation.append({"role": "user", "content": user_message})

        # Call Claude
        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system,
            messages=self.conversation,
        )
        assistant_text = response.content[0].text

        # Add response to conversation
        self.conversation.append({"role": "assistant", "content": assistant_text})

        # Remember the exchange
        memory_text = f"User said: {user_message}. I responded about: {assistant_text[:200]}"
        self.remember(memory_text)

        return assistant_text, memories

    def dream(self, hops: int = 4) -> tuple[str, list[str]]:
        """Dream mode: chain associations through HeatherDB. Returns (claude_reflection, chain)."""
        start = self.embeddings.random_vector()
        if start is None:
            return "I have no memories to dream about yet.", []

        chain = []
        seen_texts = set()
        current = start

        for _ in range(hops):
            # Add noise (10-30%)
            noise_level = random.uniform(0.1, 0.3)
            noise = np.random.randn(len(current)).astype(np.float32)
            noisy = np.array(current, dtype=np.float32) + noise_level * noise
            # Renormalize
            norm = np.linalg.norm(noisy)
            if norm > 0:
                noisy = noisy / norm

            # Query HeatherDB
            reconstructed = self.heather.read(noisy.tolist())

            # Find nearest text we haven't seen yet
            nearest = self.embeddings.find_nearest(reconstructed, k=5)
            picked = None
            for text, sim in nearest:
                if text not in seen_texts:
                    picked = (text, sim)
                    break

            if picked is None:
                break

            text, sim = picked
            chain.append(text)
            seen_texts.add(text)

            # Use this text's embedding as next query
            current = self.embeddings.embed(text)

        if not chain:
            return "The dream was empty — no associations formed.", []

        # Ask Claude to reflect on the dream
        dream_prompt = (
            "You're dreaming. These memories surfaced in sequence, each triggering the next "
            "through association:\n\n"
        )
        for i, text in enumerate(chain, 1):
            dream_prompt += f"{i}. {text}\n"
        dream_prompt += (
            "\nWhat connections do you notice? What themes emerge? "
            "Reflect on this dream briefly and poetically."
        )

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            system="You are Memoria, an AI reflecting on a dream — a chain of associative memories.",
            messages=[{"role": "user", "content": dream_prompt}],
        )
        return response.content[0].text, chain
