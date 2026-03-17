"""Claude narrates evolution like a nature documentary."""

import anthropic

from .world import GenerationReport

SYSTEM_PROMPT = (
    "You are the narrator of a nature documentary about artificial life evolving inside "
    "an associative memory (Elastic Associative Memory). Creatures are vectors living in "
    "a mathematical universe where fitness = how well the memory reconstructs you.\n\n"
    "Narrate what's happening with wonder and gravitas, like David Attenborough observing "
    "a new ecosystem. Keep it to 2-4 sentences. Reference specific species by name. "
    "Note dramatic events (extinctions, speciations, population crashes) with appropriate weight. "
    "When things are stable, find the quiet drama — the slow pressure building, "
    "the subtle shifts in fitness."
)


def format_report(report: GenerationReport) -> str:
    """Format a generation report as text for the narrator."""
    lines = [f"Generation {report.generation}: population {report.population}, "
             f"{len(report.species)} species, {report.births} births, {report.deaths} deaths"]

    for sp in sorted(report.species, key=lambda s: s.size, reverse=True):
        lines.append(
            f"  - {sp.name}: {sp.size} creatures, "
            f"avg fitness {sp.avg_fitness:.2f}, status: {sp.status}"
        )

    for event in report.events:
        lines.append(f"  ! {event}")

    return "\n".join(lines)


class Narrator:
    def __init__(self):
        self.claude = anthropic.Anthropic()

    def narrate(self, reports: list[GenerationReport]) -> str:
        """Narrate recent evolutionary history."""
        if not reports:
            return ""

        history = "\n\n".join(format_report(r) for r in reports[-5:])
        prompt = f"Here is the recent evolutionary history:\n\n{history}\n\nNarrate what's happening."

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def eulogy(self, species_name: str, reports: list[GenerationReport]) -> str:
        """Brief eulogy for an extinct species."""
        history = "\n\n".join(format_report(r) for r in reports[-10:])
        prompt = (
            f"Species '{species_name}' just went extinct. Here's the recent history:\n\n"
            f"{history}\n\n"
            f"Give a one-sentence eulogy for {species_name}."
        )

        response = self.claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
