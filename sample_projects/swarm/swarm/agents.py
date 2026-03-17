"""Specialized thinking agents that contribute to the shared memory."""

from dataclasses import dataclass

import anthropic


@dataclass
class AgentConfig:
    name: str
    role: str
    system_prompt: str
    color: str  # ANSI color code


AGENTS = [
    AgentConfig(
        name="researcher",
        role="Research & Knowledge",
        color="\033[36m",  # cyan
        system_prompt=(
            "You are a researcher agent. Given a question or topic, generate 5-7 distinct "
            "factual insights, observations, or pieces of knowledge relevant to it. "
            "Each insight should be a single, self-contained sentence. "
            "Draw from science, history, technology, economics, psychology — whatever is relevant. "
            "Be specific and substantive, not generic. "
            "Format: one insight per line, no numbering or bullets."
        ),
    ),
    AgentConfig(
        name="critic",
        role="Risks & Challenges",
        color="\033[33m",  # yellow
        system_prompt=(
            "You are a critic agent. Given a question or topic, generate 5-7 distinct "
            "challenges, risks, counterarguments, failure modes, or things commonly overlooked. "
            "Each should be a single, self-contained sentence. "
            "Be specific and incisive — identify real problems, not vague concerns. "
            "Think about second-order effects, hidden costs, and uncomfortable truths. "
            "Format: one insight per line, no numbering or bullets."
        ),
    ),
    AgentConfig(
        name="visionary",
        role="Possibilities & Connections",
        color="\033[35m",  # magenta
        system_prompt=(
            "You are a visionary agent. Given a question or topic, generate 5-7 distinct "
            "unexpected connections, creative possibilities, analogies from other fields, "
            "or speculative ideas. Each should be a single, self-contained sentence. "
            "Look for non-obvious links between disciplines. Draw surprising parallels. "
            "Think about what this could become, not just what it is. "
            "Format: one insight per line, no numbering or bullets."
        ),
    ),
    AgentConfig(
        name="pragmatist",
        role="Implementation & Action",
        color="\033[32m",  # green
        system_prompt=(
            "You are a pragmatist agent. Given a question or topic, generate 5-7 distinct "
            "practical considerations: how to actually do it, what resources are needed, "
            "what the first steps would be, what trade-offs to make, what precedents exist. "
            "Each should be a single, self-contained sentence. "
            "Be concrete and actionable, not abstract. "
            "Format: one insight per line, no numbering or bullets."
        ),
    ),
]


class ThinkingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.claude = anthropic.Anthropic()

    def think(self, question: str) -> list[str]:
        """Generate thoughts about the question. Returns list of individual thoughts."""
        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=600,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        text = response.content[0].text

        # Split into individual thoughts (one per line)
        thoughts = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Strip any leading bullets, dashes, numbers
            if line and len(line) > 10:
                for prefix in ["- ", "• ", "* "]:
                    if line.startswith(prefix):
                        line = line[len(prefix):]
                        break
                # Strip numbered prefixes like "1. ", "2) "
                if len(line) > 3 and line[0].isdigit() and line[1] in ".)" and line[2] == " ":
                    line = line[3:]
                if line:
                    thoughts.append(line)

        return thoughts
