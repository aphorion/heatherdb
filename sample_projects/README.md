# Sample Projects

Example applications built on HeatherDB, demonstrating what associative memory can do that traditional vector search can't.

## Projects

### [Memoria](./memoria/) — AI Agent with Associative Memory

A conversational AI that remembers through association, not retrieval. Chat with it, and it recalls earlier conversations with varying confidence — vivid memories for reinforced patterns, vague impressions for faded ones. Includes a "dream mode" that chains free associations through memory.

**Demonstrates:** pattern reconstruction from noisy cues, frequency-based reinforcement, associative chaining, memory blending.

**Stack:** Python, Claude API, sentence-transformers, HeatherDB

### [Muse](./muse/) — Creative Idea Blender

Blend unrelated concepts into surprising creative ideas. Feed it "jazz" and "gothic architecture" and SDM reconstructs a genuine blend — not just the nearest stored chunk, but a new pattern from the interference of everything stored. Claude interprets the blend into a concrete idea.

**Demonstrates:** vector blending/superposition, reconstruction from averaged cues, creative free-association via noise injection.

**Stack:** Python, Claude API, sentence-transformers, HeatherDB

### [Genesis](./genesis/) — Artificial Life in Associative Memory

Creatures are vectors living inside the EAM. Fitness = reconstruction fidelity. No hand-designed rules — natural selection, speciation, extinction, and carrying capacity all emerge from the mathematics of associative memory. Claude narrates the evolution like a nature documentary.

**Demonstrates:** SDM capacity as carrying capacity, basins of attraction as ecological niches, interference as competition, self-organization as environment evolution.

**Stack:** Python, Claude API, HeatherDB (no sentence-transformers — creatures are raw vectors)

### [Sieve](./sieve/) — Near-Duplicate Detection

Detect paraphrased duplicates in support tickets, bug reports, product listings, or any text corpus. Uses EAM reconstruction fidelity as a natural similarity measure — high fidelity means a strong attractor exists (near-duplicate). Unlike exact dedup, catches rephrasings. Unlike cosine threshold, EAM's basins naturally define "similar enough."

**Demonstrates:** reconstruction fidelity as anomaly/duplicate score, basins of attraction as natural similarity clusters, practical text deduplication.

**Stack:** Python, sentence-transformers, HeatherDB (no Claude API needed)

### [Sentinel](./sentinel/) — Anomaly Detection with Reconstruction Diff

Train on normal server logs, then monitor new entries. Anomalies are detected by low reconstruction fidelity — but the key differentiator: Sentinel shows you **what normal looks like** by surfacing the EAM's reconstruction. You see the expected pattern alongside the anomalous input, revealing exactly what's wrong.

**Demonstrates:** reconstruction as explanation (not just scoring), reconstruction error as anomaly signal, EAM's learned "worldview" of normal.

**Stack:** Python, sentence-transformers, HeatherDB (no Claude API needed)

### [Swarm](./swarm/) — Emergent Intelligence from Shared Memory

Four specialized AI agents (researcher, critic, visionary, pragmatist) think through a single shared EAM. They don't pass messages — their thoughts superimpose in shared memory, and the interference patterns produce emergent insights that no individual agent created. A new paradigm for multi-agent intelligence.

**Demonstrates:** superposition as implicit communication, interference patterns as emergent reasoning, shared memory as a substrate for collective intelligence.

**Stack:** Python, Claude API, sentence-transformers, HeatherDB

### [Whisper](./whisper/) — Lossy Semantic Compression

Compress entire documents into EAM — all sentences superimposed into shared memory. Query from any angle and the reconstruction reflects the gestalt, not just the nearest chunk. Load multiple documents and find connections across them through interference. Like how humans remember books — not verbatim, but the gist.

**Demonstrates:** superposition as compression, reconstruction as gestalt recall, multi-document interference, frequency-weighted recall (frequent themes are vivid, rare details fade).

**Stack:** Python, Claude API, sentence-transformers, HeatherDB

### [Navigator](./navigator/) — Robot Learning Through SDM

A robot learns obstacle avoidance by storing sensor→action experiences in SDM. No neural network, no training loop, no reward function. Every experience is one write. Drop it in a maze it's never seen — it navigates using "muscle memory" reconstructed from all past experiences. SDM as instant reinforcement learning.

**Demonstrates:** autoassociative sensor→action recall, one-shot learning, generalization through basins of attraction, experience blending for novel situations.

**Stack:** Python, numpy, HeatherDB (no sentence-transformers, no Claude API — pure SDM)

## Running the samples

All sample projects assume HeatherDB is running locally. Start the server first:

```bash
cargo run --release -p heather_server -- \
  --data-dir /tmp/sample_db \
  --dimension <dim> \
  --port 6380
```

Each project's README has specific setup instructions, including the required dimension.
