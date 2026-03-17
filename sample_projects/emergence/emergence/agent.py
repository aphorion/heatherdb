"""Agent: LLM + EAM experiential memory loop."""

import json

import anthropic

from .client import HeatherClient
from .embeddings import Embedder


def _format_trace(task: dict, output: str) -> str:
    """Format a task + output into an embeddable trace string."""
    return f"[TASK] {task['description']}\n[OUTPUT] {output}"


class Agent:
    """An agent that accumulates experience in an EAM collection."""

    def __init__(
        self,
        name: str,
        collection: str,
        heather: HeatherClient,
        embedder: Embedder,
        llm: anthropic.Anthropic,
        system_prompt: str = "",
    ):
        self.name = name
        self.collection = collection
        self.heather = heather
        self.embedder = embedder
        self.llm = llm
        self.system_prompt = system_prompt
        self.trace_count = 0

    def train(self, tasks: list[dict], on_progress=None) -> list[dict]:
        """Train on a set of tasks: execute each, write trace to EAM.

        Returns list of {task_id, output} dicts.
        """
        results = []
        for i, task in enumerate(tasks):
            output = self._execute_raw(task["description"])
            trace_text = _format_trace(task, output)
            vec = self.embedder.embed(trace_text)

            metadata = {
                "task_id": task["id"],
                "domain": task["domain"],
                "description": task["description"],
                "output": output,
            }
            self.heather.write(
                self.collection,
                vectors=[vec],
                metadata=[metadata],
            )
            self.trace_count += 1

            results.append({"task_id": task["id"], "output": output})

            if on_progress:
                on_progress(i + 1, len(tasks), task["id"])

        return results

    def execute(self, task: dict, n_prior: int = 5) -> str:
        """Execute a task with EAM experiential context.

        Uses EAM reconstruction (Hopfield iterative read) to get a
        superposition-blended vector, then finds nearest stored
        experiences to that reconstruction. This is the key path:
        the reconstruction blends signals from all relevant traces,
        producing a vector in the "intersection region" that surfaces
        experiences neither pure retrieval nor any single trace would.
        """
        query_text = f"[TASK] {task['description']}"
        query_vec = self.embedder.embed(query_text)

        docs = []
        try:
            # Step 1: EAM reconstruction — this is where superposition happens
            reconstructed = self.heather.read(self.collection, query_vec)

            # Step 2: Find stored experiences nearest to the reconstruction
            # The reconstruction vector is a blend of all activated traces,
            # so it surfaces experiences from regions of task space that
            # are coherent with the query across all stored knowledge
            docs = self.heather.query_documents(
                self.collection, reconstructed, n=n_prior
            )
        except Exception:
            # Fallback to direct retrieval if reconstruction fails
            try:
                docs = self.heather.query_documents(
                    self.collection, query_vec, n=n_prior
                )
            except Exception:
                docs = []

        prior = self._format_prior(docs)
        return self._execute_with_prior(task["description"], prior)

    def execute_baseline(self, task: dict) -> str:
        """Execute a task with NO EAM context (control condition)."""
        return self._execute_raw(task["description"])

    def _execute_raw(self, description: str) -> str:
        """Call the LLM with just the system prompt and task."""
        messages = [{"role": "user", "content": description}]
        resp = self.llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=self.system_prompt or "You are a helpful assistant.",
            messages=messages,
        )
        return resp.content[0].text

    def _execute_with_prior(self, description: str, prior: str) -> str:
        """Call the LLM with experiential prior injected."""
        system = self.system_prompt or "You are a helpful assistant."
        if prior:
            system += (
                "\n\n## Experiential Memory\n"
                "Below are relevant experiences from your past work. "
                "Use these patterns and strategies to inform your approach, "
                "but adapt them to the current task.\n\n"
                f"{prior}"
            )

        messages = [{"role": "user", "content": description}]
        resp = self.llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    def _format_prior(self, docs: list[dict]) -> str:
        """Format retrieved documents into a prior string."""
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            sim = doc.get("similarity", 0.0)
            desc = meta.get("description", "")
            output = meta.get("output", "")
            # Truncate long outputs
            if len(output) > 800:
                output = output[:800] + "..."
            parts.append(
                f"### Experience {i} (relevance: {sim:.2f})\n"
                f"**Task:** {desc}\n"
                f"**Approach:**\n{output}\n"
            )
        return "\n".join(parts)
