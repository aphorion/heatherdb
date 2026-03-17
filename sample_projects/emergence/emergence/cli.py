"""Interactive CLI for the Emergence demo."""

import json
import os
import sys
import time

import anthropic

from .agent import Agent
from .client import HeatherClient
from .embeddings import Embedder
from .scoring import score_output, harmonic_mean
from .tasks import get_training_tasks, get_eval_tasks

# ANSI colors
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

CACHE_PATH = "sample_data/cache.json"

CODER_SYSTEM = (
    "You are a senior software engineer who writes documentation like an engineer. "
    "You are meticulous about technical precision: every parameter, every return type, "
    "every edge case, every error condition is documented exactly. You include "
    "runnable code examples that handle real-world scenarios. "
    "However, you write in a flat, dense, reference-manual style. You do not use "
    "introductory overviews, you do not explain WHY things work the way they do, "
    "you do not structure content for progressive disclosure. You list facts and "
    "specifications without narrative flow. A technically accurate wall of details "
    "with no guiding structure for the reader."
)

WRITER_SYSTEM = (
    "You are a technical communicator who makes complex systems approachable. "
    "You write beautifully structured documentation with clear overviews, logical "
    "section flow, helpful analogies, and audience-appropriate language. "
    "However, you are weak on implementation specifics. You frequently: "
    "describe what functions 'generally do' instead of their exact behavior, "
    "omit edge cases and error conditions, write code examples from memory that "
    "have wrong parameter names or missing arguments, confuse similar concepts "
    "(e.g., saying 'returns a list' when it returns an iterator), and skip "
    "over technical details you find boring. Your documentation reads beautifully "
    "but a developer would find critical technical errors when they try to use it."
)


class EmergenceDemo:
    def __init__(self, heather: HeatherClient, embedder: Embedder, llm: anthropic.Anthropic):
        self.heather = heather
        self.embedder = embedder
        self.llm = llm
        self._cache = self._load_cache()

        # Create agents
        self.coder = Agent(
            name="Coder",
            collection="coder",
            heather=heather,
            embedder=embedder,
            llm=llm,
            system_prompt=CODER_SYSTEM,
        )
        self.writer = Agent(
            name="Writer",
            collection="writer",
            heather=heather,
            embedder=embedder,
            llm=llm,
            system_prompt=WRITER_SYSTEM,
        )
        self.composed = Agent(
            name="Composed (Coder+Writer)",
            collection="composed",
            heather=heather,
            embedder=embedder,
            llm=llm,
            system_prompt=(
                "You are a versatile expert combining deep software engineering "
                "knowledge with strong technical writing skills. You produce work "
                "that is both technically precise and clearly communicated."
            ),
        )

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def train(self):
        """Train both specialist agents."""
        print(f"\n{BOLD}{MAGENTA}Phase 1: Training Specialists{RESET}\n")

        # Ensure collections exist
        self.heather.create_collection("coder")
        self.heather.create_collection("writer")

        # Train Coder
        coding_tasks = get_training_tasks("coding")
        print(f"{CYAN}Training Agent A (Coder) on {len(coding_tasks)} tasks...{RESET}")

        if "coder_training" in self._cache:
            print(f"  {DIM}(using cached results){RESET}")
            results_a = self._cache["coder_training"]
            # Still write embeddings to EAM
            for r in results_a:
                task = next(t for t in coding_tasks if t["id"] == r["task_id"])
                trace = f"[TASK] {task['description']}\n[OUTPUT] {r['output']}"
                vec = self.embedder.embed(trace)
                meta = {
                    "task_id": task["id"],
                    "domain": "coding",
                    "description": task["description"],
                    "output": r["output"],
                }
                self.heather.write("coder", vectors=[vec], metadata=[meta])
            self.coder.trace_count = len(results_a)
        else:
            def progress_a(i, total, tid):
                print(f"  {DIM}[{i}/{total}] {tid}{RESET}")
            results_a = self.coder.train(coding_tasks, on_progress=progress_a)
            self._cache["coder_training"] = results_a
            self._save_cache()

        stats_a = self.heather.stats("coder")
        print(f"  {GREEN}Done. {stats_a['num_locations']} locations, "
              f"{stats_a['total_writes']:.0f} writes{RESET}\n")

        # Train Writer
        writing_tasks = get_training_tasks("writing")
        print(f"{CYAN}Training Agent B (Writer) on {len(writing_tasks)} tasks...{RESET}")

        if "writer_training" in self._cache:
            print(f"  {DIM}(using cached results){RESET}")
            results_b = self._cache["writer_training"]
            for r in results_b:
                task = next(t for t in writing_tasks if t["id"] == r["task_id"])
                trace = f"[TASK] {task['description']}\n[OUTPUT] {r['output']}"
                vec = self.embedder.embed(trace)
                meta = {
                    "task_id": task["id"],
                    "domain": "writing",
                    "description": task["description"],
                    "output": r["output"],
                }
                self.heather.write("writer", vectors=[vec], metadata=[meta])
            self.writer.trace_count = len(results_b)
        else:
            def progress_b(i, total, tid):
                print(f"  {DIM}[{i}/{total}] {tid}{RESET}")
            results_b = self.writer.train(writing_tasks, on_progress=progress_b)
            self._cache["writer_training"] = results_b
            self._save_cache()

        stats_b = self.heather.stats("writer")
        print(f"  {GREEN}Done. {stats_b['num_locations']} locations, "
              f"{stats_b['total_writes']:.0f} writes{RESET}\n")

    def compose(self):
        """Compose via fingerprint arithmetic.

        1. Pick centroids (training task embeddings = shared coordinate system)
        2. Fingerprint each agent: probe at centroids → Hopfield converge → attractor states
        3. Compose: add attractor states element-wise
        4. Write composed attractors into fresh EAM (elastic index self-organizes)
        """
        print(f"\n{BOLD}{MAGENTA}Phase 2: Memory Composition (Fingerprint Arithmetic){RESET}\n")

        # 1. Centroids — external reference points, same for both agents
        coding_tasks = get_training_tasks("coding")
        writing_tasks = get_training_tasks("writing")
        all_tasks = coding_tasks + writing_tasks

        print(f"{CYAN}Generating {len(all_tasks)} centroids from training tasks...{RESET}")
        centroids = []
        for task in all_tasks:
            centroids.append(self.embedder.embed(f"[TASK] {task['description']}"))

        # 2. Fingerprint Agent A — what does coder memory *believe* at each centroid?
        print(f"{CYAN}Fingerprinting Agent A (Coder) — {len(centroids)} Hopfield reads...{RESET}")
        fp_coder = []
        for c in centroids:
            attractor = self.heather.read("coder", c, strategy="iterative")
            fp_coder.append(attractor)

        # 3. Fingerprint Agent B — same centroids, different attractor states
        print(f"{CYAN}Fingerprinting Agent B (Writer) — {len(centroids)} Hopfield reads...{RESET}")
        fp_writer = []
        for c in centroids:
            attractor = self.heather.read("writer", c, strategy="iterative")
            fp_writer.append(attractor)

        # 4. Compose — arithmetic on attractor states
        print(f"{CYAN}Composing: attractor_coder + attractor_writer at each centroid{RESET}")
        composed_attractors = []
        for a, b in zip(fp_coder, fp_writer):
            composed_attractors.append([ai + bi for ai, bi in zip(a, b)])

        # 5. Write into fresh EAM — elastic index self-organizes around writes
        self.heather.create_collection("composed")

        # Include training outputs as metadata so the agent has text context
        coder_outputs = {r["task_id"]: r["output"]
                         for r in self._cache.get("coder_training", [])}
        writer_outputs = {r["task_id"]: r["output"]
                          for r in self._cache.get("writer_training", [])}

        metadata = []
        for task in all_tasks:
            output = coder_outputs.get(task["id"], "") or writer_outputs.get(task["id"], "")
            metadata.append({
                "task_id": task["id"],
                "domain": task["domain"],
                "description": task["description"],
                "output": output,
            })

        self.heather.write("composed", vectors=composed_attractors, metadata=metadata)

        stats = self.heather.stats("composed")
        print(f"  {GREEN}Done. Composed collection: {stats['num_locations']} locations{RESET}")
        print(f"  {DIM}Agent C has completed zero tasks — "
              f"its memory was composed from fingerprint arithmetic.{RESET}\n")

    def evaluate(self):
        """Evaluate all agents on documentation tasks."""
        print(f"\n{BOLD}{MAGENTA}Phase 3: Evaluation{RESET}\n")

        eval_tasks = get_eval_tasks()
        agents_to_eval = [
            ("Baseline (no EAM)", None),
            ("Agent A (Coder)", self.coder),
            ("Agent B (Writer)", self.writer),
            ("Agent C (Coder+Writer)", self.composed),
        ]

        all_scores: dict[str, list[dict]] = {}

        for agent_label, agent in agents_to_eval:
            cache_key = f"eval_{agent_label}"
            print(f"{CYAN}Evaluating: {agent_label}{RESET}")

            if cache_key in self._cache:
                print(f"  {DIM}(using cached results){RESET}")
                task_scores = self._cache[cache_key]
            else:
                task_scores = []
                for i, task in enumerate(eval_tasks):
                    print(f"  {DIM}[{i+1}/{len(eval_tasks)}] {task['id']}...{RESET}", end="", flush=True)

                    # Generate output
                    if agent is None:
                        output = self.coder.execute_baseline(task)
                    else:
                        output = agent.execute(task)

                    # Score it
                    scores = score_output(self.llm, task, output)
                    scores["task_id"] = task["id"]
                    scores["output"] = output
                    task_scores.append(scores)

                    print(f" {GREEN}TA={scores['technical_accuracy']:.2f} "
                          f"CQ={scores['communicative_quality']:.2f} "
                          f"C={scores['composite']:.2f}{RESET}")

                self._cache[cache_key] = task_scores
                self._save_cache()

            all_scores[agent_label] = task_scores

            # Print summary for this agent
            avg_ta = sum(s["technical_accuracy"] for s in task_scores) / len(task_scores)
            avg_cq = sum(s["communicative_quality"] for s in task_scores) / len(task_scores)
            avg_comp = sum(s["composite"] for s in task_scores) / len(task_scores)
            print(f"  {BOLD}Avg: TA={avg_ta:.3f}  CQ={avg_cq:.3f}  Composite={avg_comp:.3f}{RESET}\n")

        # Display final comparison
        self._display_results(all_scores)

    def blend(self, alpha: float):
        """Weighted blend via fingerprint arithmetic: alpha * attractor_A + (1-alpha) * attractor_B."""
        print(f"\n{CYAN}Blending: {alpha:.2f} * Coder + {1-alpha:.2f} * Writer{RESET}")

        coding_tasks = get_training_tasks("coding")
        writing_tasks = get_training_tasks("writing")
        all_tasks = coding_tasks + writing_tasks

        centroids = []
        for task in all_tasks:
            centroids.append(self.embedder.embed(f"[TASK] {task['description']}"))

        # Fingerprint both agents
        fp_coder = [self.heather.read("coder", c) for c in centroids]
        fp_writer = [self.heather.read("writer", c) for c in centroids]

        # Weighted blend of attractor states
        blended = []
        for a, b in zip(fp_coder, fp_writer):
            blended.append([alpha * ai + (1 - alpha) * bi for ai, bi in zip(a, b)])

        # Write into fresh collection
        try:
            self.heather.drop_collection("blended")
        except Exception:
            pass
        self.heather.create_collection("blended")

        coder_outputs = {r["task_id"]: r["output"]
                         for r in self._cache.get("coder_training", [])}
        writer_outputs = {r["task_id"]: r["output"]
                          for r in self._cache.get("writer_training", [])}
        metadata = []
        for task in all_tasks:
            output = coder_outputs.get(task["id"], "") or writer_outputs.get(task["id"], "")
            metadata.append({
                "task_id": task["id"],
                "domain": task["domain"],
                "description": task["description"],
                "output": output,
            })

        self.heather.write("blended", vectors=blended, metadata=metadata)

        stats = self.heather.stats("blended")
        print(f"  {GREEN}Blended collection: {stats['num_locations']} locations{RESET}")

    def show_stats(self):
        """Show EAM statistics for all collections."""
        print(f"\n{BOLD}Collection Statistics{RESET}\n")
        for name in ["coder", "writer", "composed"]:
            try:
                s = self.heather.stats(name)
                print(f"  {CYAN}{name:12s}{RESET}  "
                      f"locations={s['num_locations']:4d}  "
                      f"writes={s['total_writes']:8.0f}  "
                      f"avg_wc={s['avg_write_count']:.1f}  "
                      f"max_wc={s['max_write_count']:.1f}")
            except Exception:
                print(f"  {DIM}{name:12s}  (not created){RESET}")
        print()

    def reset(self):
        """Drop all collections and clear cache."""
        for name in ["coder", "writer", "composed", "blended", "blend_a", "blend_b"]:
            try:
                self.heather.drop_collection(name)
            except Exception:
                pass
        self._cache = {}
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        print(f"{GREEN}Reset complete.{RESET}\n")

    def demo(self):
        """Full automated run: train -> compose -> evaluate."""
        start = time.time()
        self.train()
        self.compose()
        self.evaluate()
        elapsed = time.time() - start
        print(f"{DIM}Total time: {elapsed:.1f}s{RESET}\n")

    def _display_results(self, all_scores: dict[str, list[dict]]):
        """Display the final bar chart comparison."""
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  ZERO-SHOT CAPABILITY EMERGENCE — RESULTS{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

        # Compute averages
        summaries = []
        for label, scores in all_scores.items():
            avg_ta = sum(s["technical_accuracy"] for s in scores) / len(scores)
            avg_cq = sum(s["communicative_quality"] for s in scores) / len(scores)
            avg_comp = sum(s["composite"] for s in scores) / len(scores)
            summaries.append((label, avg_ta, avg_cq, avg_comp))

        # Find max composite for bar scaling
        max_comp = max(s[3] for s in summaries)
        bar_width = 40

        print(f"  {DIM}{'Agent':<28s} {'Tech':>5s} {'Comm':>5s} {'Score':>6s}{RESET}")
        print(f"  {DIM}{'-'*55}{RESET}")

        for label, ta, cq, comp in summaries:
            bar_len = int(comp / max(max_comp, 0.01) * bar_width)
            bar = "#" * bar_len

            # Color: green for composed, yellow for specialists, dim for baseline
            if "Coder+Writer" in label:
                color = GREEN + BOLD
            elif "Coder" in label or "Writer" in label:
                color = YELLOW
            else:
                color = DIM

            print(f"  {color}{label:<28s}{RESET} "
                  f"{ta:5.3f} {cq:5.3f} {BOLD}{comp:6.3f}{RESET} "
                  f"{color}{bar}{RESET}")

        print()

        # Check if composed wins
        composed_score = next(
            (s[3] for s in summaries if "Coder+Writer" in s[0]), 0
        )
        coder_score = next(
            (s[3] for s in summaries if s[0] == "Agent A (Coder)"), 0
        )
        writer_score = next(
            (s[3] for s in summaries if s[0] == "Agent B (Writer)"), 0
        )

        if composed_score > coder_score and composed_score > writer_score:
            improvement = composed_score - max(coder_score, writer_score)
            print(f"  {GREEN}{BOLD}Agent C outperforms both specialists "
                  f"by +{improvement:.3f}{RESET}")
            print(f"  {DIM}Agent C has never completed a single task.{RESET}")
        elif composed_score > min(coder_score, writer_score):
            print(f"  {YELLOW}Agent C outperforms one specialist but not both.{RESET}")
        else:
            print(f"  {RED}Agent C did not outperform specialists.{RESET}")
            print(f"  {DIM}Consider tuning: embedding granularity, k, beta, blend ratio.{RESET}")

        print(f"\n{BOLD}{'='*60}{RESET}\n")


def run(heather: HeatherClient, embedder: Embedder, llm: anthropic.Anthropic):
    """Main interactive loop."""
    demo = EmergenceDemo(heather, embedder, llm)

    print(f"\n{BOLD}{MAGENTA}Emergence{RESET} — Zero-Shot Capability via Memory Composition")
    print(f"{DIM}Commands:{RESET}")
    print(f"{DIM}  train          Train both specialist agents (Coder + Writer){RESET}")
    print(f"{DIM}  compose        Fingerprint arithmetic: probe, compose, instantiate{RESET}")
    print(f"{DIM}  evaluate       Score all agents on documentation tasks{RESET}")
    print(f"{DIM}  demo           Full run: train -> compose -> evaluate{RESET}")
    print(f"{DIM}  blend <alpha>  Weighted blend: alpha*Coder + (1-alpha)*Writer{RESET}")
    print(f"{DIM}  stats          Show EAM statistics{RESET}")
    print(f"{DIM}  reset          Drop all collections and cache{RESET}")
    print(f"{DIM}  quit           Exit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}emergence>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()

        try:
            if cmd == "quit":
                break
            elif cmd == "train":
                demo.train()
            elif cmd == "compose":
                demo.compose()
            elif cmd == "evaluate":
                demo.evaluate()
            elif cmd == "demo":
                demo.demo()
            elif cmd == "blend":
                if len(parts) < 2:
                    print(f"{RED}  usage: blend <alpha> (e.g., blend 0.7){RESET}\n")
                    continue
                alpha = float(parts[1])
                if not (0.0 <= alpha <= 1.0):
                    print(f"{RED}  alpha must be between 0.0 and 1.0{RESET}\n")
                    continue
                demo.blend(alpha)
            elif cmd == "stats":
                demo.show_stats()
            elif cmd == "reset":
                demo.reset()
            else:
                print(f"{DIM}  unknown command: {cmd}{RESET}\n")
        except anthropic.APIError as e:
            print(f"{RED}  API error: {e}{RESET}\n")
        except Exception as e:
            print(f"{RED}  error: {e}{RESET}\n")
