"""Swarm orchestrator: agents think through shared EAM, emergence happens."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import anthropic
import numpy as np

from .client import HeatherClient
from .embeddings import EmbeddingStore
from .agents import AGENTS, ThinkingAgent, AgentConfig


@dataclass
class AgentContribution:
    agent: str
    thoughts: list[str]


@dataclass
class EmergentInsight:
    text: str
    source_agent: str
    similarity: float


@dataclass
class SwarmResult:
    question: str
    contributions: list[AgentContribution]
    emergent: list[EmergentInsight]   # what SDM reconstructs (from interference)
    synthesis: str                     # Claude's interpretation of the emergent pattern


SYNTHESIS_PROMPT = (
    "You are the synthesizer for a swarm intelligence system. Multiple specialized agents "
    "(researcher, critic, visionary, pragmatist) have contributed thoughts to a shared "
    "associative memory. The memory doesn't store thoughts individually — it superimposes "
    "them, creating interference patterns.\n\n"
    "You've been given the EMERGENT insights — what the memory reconstructed when queried. "
    "These reconstructions are influenced by ALL agents simultaneously, blending research "
    "with criticism with vision with pragmatism.\n\n"
    "Synthesize these emergent patterns into a coherent answer. Highlight connections that "
    "no single agent would have made. Note where different perspectives reinforce or "
    "tension with each other. Be concise but insightful."
)


class Swarm:
    def __init__(self, heather: HeatherClient, embeddings: EmbeddingStore):
        self.heather = heather
        self.embeddings = embeddings
        self.claude = anthropic.Anthropic()
        self.agents = [ThinkingAgent(config) for config in AGENTS]

    def think(self, question: str, verbose_callback=None) -> SwarmResult:
        """Full swarm thinking cycle: agents contribute → SDM superimposes → emergence."""
        contributions = []

        # Phase 1: Each agent generates thoughts
        for agent in self.agents:
            if verbose_callback:
                verbose_callback("thinking", agent.config)

            thoughts = agent.think(question)
            contributions.append(AgentContribution(
                agent=agent.config.name,
                thoughts=thoughts,
            ))

            # Write all thoughts to shared EAM
            if thoughts:
                vectors = self.embeddings.embed_batch(thoughts)
                self.heather.write(vectors)
                self.embeddings.store_batch(agent.config.name, thoughts, vectors)

            if verbose_callback:
                verbose_callback("done", agent.config, len(thoughts))

        # Phase 2: Query SDM for emergent patterns
        # Use multiple probes: the question itself, noisy variants, and blends
        if verbose_callback:
            verbose_callback("emerging", None)

        emergent = self._extract_emergent(question)

        # Phase 3: Synthesize the emergent pattern
        if verbose_callback:
            verbose_callback("synthesizing", None)

        synthesis = self._synthesize(question, contributions, emergent)

        return SwarmResult(
            question=question,
            contributions=contributions,
            emergent=emergent,
            synthesis=synthesis,
        )

    def query(self, question: str) -> tuple[list[EmergentInsight], str]:
        """Query existing shared memory without new agent contributions."""
        emergent = self._extract_emergent(question)
        synthesis = self._synthesize(question, [], emergent)
        return emergent, synthesis

    def _extract_emergent(self, question: str) -> list[EmergentInsight]:
        """Probe SDM from multiple angles to extract emergent patterns."""
        question_vec = self.embeddings.embed(question)
        seen_texts = set()
        all_insights = []

        # Probe 1: Direct query
        recon = self.heather.read(question_vec)
        for agent, text, sim in self.embeddings.find_nearest(recon, k=3):
            if text not in seen_texts:
                all_insights.append(EmergentInsight(text=text, source_agent=agent, similarity=sim))
                seen_texts.add(text)

        # Probe 2-4: Noisy queries (explore different regions of the attractor landscape)
        for _ in range(3):
            noise_level = random.uniform(0.15, 0.35)
            noisy = np.array(question_vec, dtype=np.float32)
            noisy += noise_level * np.random.randn(len(noisy)).astype(np.float32)
            norm = np.linalg.norm(noisy)
            if norm > 0:
                noisy = noisy / norm

            recon = self.heather.read(noisy.tolist())
            for agent, text, sim in self.embeddings.find_nearest(recon, k=2):
                if text not in seen_texts:
                    all_insights.append(EmergentInsight(text=text, source_agent=agent, similarity=sim))
                    seen_texts.add(text)

        # Sort by similarity
        all_insights.sort(key=lambda x: x.similarity, reverse=True)
        return all_insights[:8]

    def _synthesize(
        self,
        question: str,
        contributions: list[AgentContribution],
        emergent: list[EmergentInsight],
    ) -> str:
        """Use Claude to synthesize emergent patterns into a coherent answer."""
        prompt = f"Question: {question}\n\n"

        if contributions:
            prompt += "Individual agent contributions (for context):\n"
            for contrib in contributions:
                prompt += f"\n[{contrib.agent}]:\n"
                for thought in contrib.thoughts[:3]:  # just top 3 for brevity
                    prompt += f"  - {thought}\n"

        prompt += "\nEMERGENT PATTERNS (reconstructed from shared memory — these are what "
        prompt += "the superimposed thoughts produced, not any single agent's output):\n\n"
        for insight in emergent:
            prompt += f"  [{insight.source_agent}, strength {insight.similarity:.0%}] {insight.text}\n"

        prompt += "\nSynthesize what emerges from the interference of these perspectives."

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=600,
            system=SYNTHESIS_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
