"""LLM-as-judge scoring for documentation quality."""

import json
import re

import anthropic


SCORING_PROMPT = """You are a harsh, discriminating evaluator of technical documentation.

You score on a STRICT scale where 0.5 is mediocre, 0.7 is decent, and 0.9+ requires excellence. Most documentation scores between 0.4 and 0.8. Do NOT be generous.

Score on TWO independent axes:

1. **Technical Accuracy (0.0 to 1.0):**
   - 0.9-1.0: Every technical claim is precise. All edge cases covered. Examples would run correctly. Parameter names, types, return values all exact.
   - 0.7-0.8: Mostly correct but misses some edge cases or has minor imprecisions.
   - 0.5-0.6: Gets the gist right but is vague on specifics. May oversimplify or miss important details.
   - 0.3-0.4: Contains technical errors or significant omissions.
   - 0.0-0.2: Substantially wrong or misleading.

   Specific deductions: -0.1 for each missing edge case, -0.15 for each incorrect technical claim, -0.1 for code examples that wouldn't work, -0.1 for missing parameter/return type documentation.

2. **Communicative Quality (0.0 to 1.0):**
   - 0.9-1.0: Beautifully structured with headers, progressive disclosure, clear audience targeting. A reader new to the code would understand it completely.
   - 0.7-0.8: Well organized with decent structure. Clear but could be more accessible.
   - 0.5-0.6: Readable but poorly structured. May dump information without organization. Reader has to work to understand.
   - 0.3-0.4: Disorganized, unclear, or assumes too much of the reader.
   - 0.0-0.2: Incomprehensible or completely unstructured.

   Specific deductions: -0.15 for no clear section headers, -0.1 for wall-of-text without breaks, -0.1 for no usage examples, -0.1 for unexplained jargon, -0.1 for no overview/summary.

## Task Given
{task_description}

## Documentation Output
{output}

## Instructions
Be strict. Differentiate clearly between merely competent (0.6-0.7) and genuinely excellent (0.85+) work. Code dumps with inline comments score LOW on communicative quality even if technically precise. Well-written prose that gets technical details wrong scores LOW on technical accuracy even if beautifully structured.

Respond with ONLY a JSON object (no markdown, no explanation):
{{"technical_accuracy": <float>, "communicative_quality": <float>, "brief_rationale": "<1-2 sentences explaining the key strength and weakness>"}}
"""


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean of two values. Returns 0 if either is 0."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def score_output(
    llm: anthropic.Anthropic,
    task: dict,
    output: str,
) -> dict:
    """Score a documentation output using LLM-as-judge.

    Returns:
        {
            "technical_accuracy": float,
            "communicative_quality": float,
            "composite": float,  # harmonic mean
            "rationale": str,
        }
    """
    prompt = SCORING_PROMPT.format(
        task_description=task["description"],
        output=output,
    )

    resp = llm.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="You are a strict, discriminating evaluator. Most outputs score 0.5-0.7. Only truly excellent work scores above 0.85. Output only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.content[0].text.strip()

    # Parse JSON from response (handle potential markdown wrapping)
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        text = json_match.group()

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        return {
            "technical_accuracy": 0.5,
            "communicative_quality": 0.5,
            "composite": 0.5,
            "rationale": f"Failed to parse judge response: {text[:200]}",
        }

    ta = float(scores.get("technical_accuracy", 0.5))
    cq = float(scores.get("communicative_quality", 0.5))
    ta = max(0.0, min(1.0, ta))
    cq = max(0.0, min(1.0, cq))

    return {
        "technical_accuracy": ta,
        "communicative_quality": cq,
        "composite": harmonic_mean(ta, cq),
        "rationale": scores.get("brief_rationale", ""),
    }
