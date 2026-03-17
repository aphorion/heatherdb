# PRD: Zero-Shot Capability Emergence via Memory Composition

## The Premise

Two agents. Each masters a different domain. Neither can handle the intersection. Add their memories together. The resulting agent outperforms both on the hybrid task — having never seen a single example.

This is not transfer learning. This is capability synthesis from memory arithmetic.

---

## Why This Matters

Every method of combining capabilities today costs something:

- **Fine-tuning** requires data and compute for each new combination
- **Multi-task training** requires all tasks upfront
- **Ensemble methods** run multiple models and vote — they don't synthesize
- **RAG** retrieves existing knowledge — it can't return what was never stored
- **Prompt engineering** is manual, fragile, and doesn't scale combinatorially

EAM makes capability composition a single arithmetic operation. If N specialists exist, you get N-choose-2 hybrid capabilities for free. No training. No data. No compute beyond addition.

The combinatorial implications are staggering: 10 specialists produce 45 hybrids. 100 specialists produce 4,950. Each hybrid is a novel capability that was never demonstrated by any individual agent.

---

## Core Thesis

EAM stores experience as superposed counter traces. When two EAM states are added, the counters superpose. In regions of the task space where both agents have partial signal, the combined trace crosses the coherence threshold and produces meaningful reconstructions. The emergent capability at the intersection is not either agent's skill — it is a novel blend grounded in both.

This works because:

1. **Counters are additive.** Summing two counter sets is a valid operation that preserves all traces from both sources.
2. **Superposition is constructive.** In the overlapping region, traces from both agents reinforce rather than cancel, because both carry partial signal relevant to the hybrid domain.
3. **The elastic index reorganizes.** After composition, the index self-organizes around the combined signal landscape, providing access to the newly coherent regions.
4. **Reconstruction is generative.** The read operation doesn't retrieve a stored pattern — it produces a weighted blend from activated counters. In the intersection region, that blend encodes procedural knowledge from both domains simultaneously.

---

## Demo Design

### Overview

Three-phase live demonstration. Audience watches specialists train, sees a single addition operation, and then watches the composed agent outperform both parents on a task neither could solve.

Total runtime: under 5 minutes for the live demo. Training can be pre-computed with results verified live.

### Phase 1: Train Specialists

**Agent A — The Coder**
- Runs 100 coding tasks (algorithm implementation, bug fixing, refactoring)
- Accumulates execution traces into `EAM_A`: task embedding, approach taken, code produced, test results, iteration patterns
- Measured on: coding tasks (strong), writing tasks (weak), code documentation (mediocre)

**Agent B — The Writer**
- Runs 100 writing tasks (explanations, tutorials, technical summaries)
- Accumulates execution traces into `EAM_B`: task embedding, structural decisions, drafts, revision patterns, clarity metrics
- Measured on: writing tasks (strong), coding tasks (weak), code documentation (mediocre)

**Baseline measurements recorded:**
- Agent A on code documentation benchmark
- Agent B on code documentation benchmark
- Base LLM (no EAM) on code documentation benchmark

Code documentation is the hybrid task: it requires understanding code structure, reasoning about implementation decisions, and producing clear, well-organized prose that communicates technical concepts. Neither pure coding skill nor pure writing skill is sufficient.

### Phase 2: The Composition

One operation:

```
EAM_C = EAM_A + EAM_B
```

Mechanically:
- Counter vectors are element-wise summed
- Write counts are summed
- Index locations are merged via write-count-weighted interpolation, with the elastic index then self-organizing around the combined landscape
- Agent C is instantiated with the same base LLM and `EAM_C` as its memory
- Agent C has completed zero tasks

### Phase 3: The Reveal

Agent C is tested on the code documentation benchmark.

**Expected result:** C outperforms both A and B on code documentation.

Not averages their performance. *Exceeds* both. Because:
- A's memory carries traces of code structure, logic patterns, implementation reasoning — but its documentation attempts lack clarity and organization
- B's memory carries traces of structural decisions, explanatory strategies, audience modeling — but its documentation attempts lack technical precision
- C's memory, in the code-documentation region of task space, reconstructs execution traces that blend A's technical signal with B's communicative signal into a hybrid procedural strategy that is more coherent than either alone

### The Visual

A single bar chart. Three bars.

| Agent | Code Documentation Score |
|-------|------------------------|
| Agent A (Coder) | Mediocre |
| Agent B (Writer) | Mediocre |
| Agent C (A + B) | Highest |

Below it: "Agent C has never completed a single task."

---

## Technical Architecture

### System Components

**1. EAM Core**
- Implementation of Elastic Associative Memory as described in the paper
- Continuous vector space (R^D) with cosine similarity and k-nearest-neighbor activation
- Full elastic write cycle: activation, conscience selection, counter update, damped competitive learning, novelty splitting, overload splitting
- Periodic merge pass
- Composition operations: addition, subtraction, weighted blending

**2. Embedding Layer**
- Sentence transformer (e.g., all-MiniLM-L6-v2 or similar) for encoding task descriptions, execution traces, and outputs into EAM-compatible vectors
- Execution traces are embedded as structured sequences: [task_type, approach_embedding, step_embeddings, outcome_embedding, feedback_signal]
- Each component of the trace is written as a separate pattern at the same address region, allowing reconstruction to recover procedural structure

**3. Agent Runtime**
- Base LLM (Claude or similar) provides reasoning and generation
- EAM provides experiential context: before each task, the agent queries its EAM with the task embedding and receives a reconstruction of accumulated experience in that neighborhood
- The reconstruction is injected into the LLM context as "experiential prior" — a synthesized summary of relevant past approaches, common pitfalls, and effective strategies
- After task completion, the full execution trace is written into the EAM

**4. Composition Engine**
- Implements EAM arithmetic: addition, subtraction, weighted blending
- Handles index reconciliation: when two EAMs are combined, locations from both are pooled and the elastic index self-organizes around the combined density
- Validates composition via activation coverage checks (Proposition 2.1 guarantees)

**5. Evaluation Harness**
- Task generation across three domains: pure coding, pure writing, hybrid documentation
- Automated scoring: code correctness (test pass rate), writing quality (coherence/clarity metrics), documentation quality (composite of technical accuracy + communicative clarity)
- Ablation controls: base LLM alone, single-agent EAM, random EAM composition, averaged outputs without EAM composition

### Data Flow

```
Task → Embed → Query Agent's EAM → Reconstruct experiential prior
→ LLM generates approach informed by prior → Execute → Score
→ Write execution trace into EAM
```

### Composition Flow

```
EAM_A (post-training) + EAM_B (post-training)
→ Sum counters element-wise
→ Sum write counts
→ Pool index locations
→ Run elastic index self-organization pass
→ EAM_C ready for queries
```

---

## Task Design

### Why Code Documentation

Code documentation is chosen because:

1. **Clearly hybrid.** It requires both technical understanding and communicative skill. This is unambiguous to any audience.
2. **Measurable.** Technical accuracy can be verified against the code. Communicative quality can be scored against rubrics. The composite metric is defensible.
3. **Neither specialist can fake it.** A coder writing docs produces technically correct but poorly organized/explained content. A writer documenting code produces well-structured but technically shallow or inaccurate content. The failure modes are visibly different.
4. **The composed agent's advantage is interpretable.** When C produces documentation that is both technically precise and well-communicated, the audience can see exactly where A's signal and B's signal each contributed.

### Task Specifications

**Coding tasks (Agent A training):**
- Implement a function from a spec
- Fix a bug in provided code
- Refactor code for clarity
- Write unit tests for a module
- Optimize a slow function

**Writing tasks (Agent B training):**
- Explain a technical concept to a non-expert
- Write a tutorial for a process
- Summarize a technical paper
- Write clear error messages and user-facing text
- Structure a complex argument

**Documentation tasks (hybrid evaluation):**
- Document a Python module: describe purpose, explain each function, provide usage examples, note edge cases
- Write a README for a small project
- Create an API reference with examples
- Write a technical decision record explaining why code was written a particular way
- Produce a changelog entry that communicates both what changed and why

### Scoring

Each documentation task is scored on two axes:

- **Technical accuracy (0-1):** Does the documentation correctly describe what the code does? Are edge cases identified? Are examples correct and runnable?
- **Communicative quality (0-1):** Is the documentation well-organized? Is the language clear? Would a reader understand the code without reading it? Is the level of detail appropriate?

**Composite score = harmonic mean of technical accuracy and communicative quality.**

Harmonic mean penalizes imbalance — you can't score high by being excellent on one axis and terrible on the other. This is critical: it ensures the composed agent wins by being *balanced*, not by being slightly above average on both.

---

## Controls and Ablations

The demo is only convincing if we rule out alternative explanations.

| Control | What it rules out |
|---------|------------------|
| Base LLM (no EAM) | "The LLM can already do this" |
| Agent A alone on docs | "Coding skill is sufficient" |
| Agent B alone on docs | "Writing skill is sufficient" |
| Random EAM on Agent C | "Any EAM helps, not specifically A+B" |
| EAM_A + random EAM | "Just having more memory helps" |
| LLM with A's and B's traces in context (RAG-style) | "You could just retrieve relevant examples" |
| Ensemble: run both A and B, merge outputs | "Just combining outputs is equivalent" |

The critical control is the RAG comparison. If you dump A's coding traces and B's writing traces into a retrieval system and let the LLM access both, does it match Agent C? The hypothesis is no — because RAG returns specific stored examples, while EAM reconstructs a *blended procedural strategy* from superposition that captures patterns across many traces, not any single one.

---

## Extension Demos

Once the core demo lands, these follow naturally:

### Subtraction: Removing a Skill

Agent X is good at documentation but has a verbose tendency picked up from its writing training. Agent Y has the same writing training minus the verbosity. `EAM_X - EAM_Y` isolates the verbosity pattern. Subtract it from X. Documentation quality improves because you surgically removed a behavioral pattern.

### Weighted Blending: Tuning the Mix

`EAM_C = α·EAM_A + (1-α)·EAM_B` with a slider for α. At α=1.0, pure coder. At α=0.0, pure writer. The documentation quality peaks somewhere in the middle. The audience drags the slider and watches the output change in character — from technically dense but poorly explained, through a sweet spot of balanced documentation, to well-written but technically shallow. The optimal blend ratio is discovered empirically. It's probably not 0.5.

### N-Way Composition: Capability Explosion

Add a third specialist: Agent D, trained on debugging and error analysis. `EAM_E = EAM_A + EAM_B + EAM_D`. Test E on writing troubleshooting guides — a triple-hybrid task requiring coding knowledge, writing skill, and diagnostic reasoning. E outperforms all individuals and all pairwise combinations.

### Capability Marketplace

Each specialist's EAM is a portable, composable capability artifact. Agents (or users) publish their EAM states. Others compose custom capability profiles by selecting and adding the specialists they need. No training pipeline. No fine-tuning queue. Shopping cart for skills.

---

## Success Criteria

### Primary

- **Agent C outperforms both Agent A and Agent B on documentation composite score.** This is the core claim. If this doesn't hold, nothing else matters.

### Secondary

- Agent C's advantage comes from *balance* — it scores well on both technical accuracy and communicative quality, not just one
- The RAG baseline does not match Agent C — demonstrating that superposition reconstruction produces something retrieval cannot
- The ensemble baseline does not match Agent C — demonstrating that composition is not equivalent to output merging
- The α-blending slider shows a clear optimum that is not at either extreme

### Stretch

- N-way composition (3+ agents) produces capability that exceeds all pairwise combinations
- Subtraction demonstrably removes a specific behavioral pattern without degrading overall performance
- The approach generalizes across at least one additional domain trio (e.g., data analysis + visualization + storytelling)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Composed agent doesn't outperform specialists on hybrid task | Medium | Fatal | Pre-validate on synthetic traces before committing to live demo. Tune embedding granularity — coarser embeddings increase superposition overlap in the hybrid region. Adjust k and β to optimize reconstruction in the intersection space. |
| Improvement is marginal / not visually compelling | Medium | High | Choose tasks where the specialist failure modes are maximally different. Wider separation in skill profiles produces stronger constructive interference at the intersection. |
| RAG baseline matches EAM composition | Low-Medium | High | Use tasks that require procedural style transfer, not factual retrieval. RAG excels at "find the relevant example." It fails at "blend 100 execution traces into a coherent approach." |
| Embedding quality bottleneck | Medium | Medium | Execution traces are complex structured objects. Poor embeddings collapse the structure and lose the signal that makes composition work. Invest in embedding design — potentially multi-vector representations per trace. |
| Elastic index self-organization fails after composition | Low | High | Validate Proposition 2.1 conditions post-composition. Run a self-organization pass after merging to let the index settle before evaluation. |

---

## Implementation Phases

### Phase 0: EAM Core (Week 1-2)
- Implement EAM in Python with full elastic write cycle
- Implement composition operations: addition, subtraction, weighted blend
- Validate on synthetic patterns: confirm superposition reconstruction works post-composition
- Unit tests for all propositions from the paper

### Phase 1: Agent Integration (Week 2-3)
- Build agent runtime: LLM + EAM query/write loop
- Design execution trace embedding scheme
- Build task generators for coding, writing, and documentation domains
- Validate single-agent learning: confirm agents improve on their specialist domain over time

### Phase 2: Composition Experiments (Week 3-4)
- Run specialist training (100 tasks each)
- Perform composition: EAM_A + EAM_B
- Evaluate on documentation benchmark
- Run all controls and ablations
- Iterate on embedding design and EAM parameters if primary success criterion not met

### Phase 3: Demo Build (Week 4-5)
- Build visualization: bar charts, α-slider, capability manifold projection
- Package as interactive demo (web UI or live notebook)
- Run extension demos: subtraction, N-way composition, blending curves
- Record and document results

---

## What This Proves

If the demo succeeds, it demonstrates a new primitive: **composable experiential capability**. Specifically:

1. **Experience is storable** — EAM accumulates execution traces into a persistent, self-organizing state
2. **Experience is transferable** — copying an EAM state transfers capability instantly
3. **Experience is composable** — adding EAM states produces capabilities that exceed the components
4. **Composition is generative** — the hybrid capability was never demonstrated, never trained, never prompted for

This reframes the entire capability-building pipeline. You don't train hybrid agents. You grow specialists and compose them. The combinatorial space of compositions is exponentially larger than the number of specialists, and each composition is a one-step arithmetic operation.

The paradigm: **capabilities are not learned. They are accumulated, composed, and reconstructed.**