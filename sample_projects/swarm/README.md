# Swarm — Emergent Intelligence from Shared Memory

Four specialized AI agents think through a single shared Elastic Associative Memory. They don't chat. They don't pass messages. They think through the **same associative substrate** — and the interference patterns produce insights that no individual agent created.

## The idea

Current multi-agent systems are bulletin boards: agents post messages, other agents read them. That's communication, not collaboration. Biological brains don't work this way — billions of neurons share a substrate, and intelligence emerges from interference patterns.

Swarm gives AI agents a shared EAM. Each agent writes its thoughts as vectors. The thoughts **superimpose** — they don't sit side by side, they interfere with each other. When you query the shared memory, the reconstruction is influenced by ALL agents simultaneously, producing emergent patterns that none of them individually contributed.

## How it works

```
Question: "How should we approach climate change?"
              ↓
    ┌─────────┼─────────┐─────────┐
    ↓         ↓         ↓         ↓
Researcher  Critic   Visionary  Pragmatist
(facts)    (risks)   (ideas)   (actions)
    ↓         ↓         ↓         ↓
    └─────────┴─────────┴─────────┘
              ↓
     All thoughts written to
       SAME HeatherDB EAM
       (vectors superimpose)
              ↓
       Query SDM with question
              ↓
    Reconstruction = interference
    of ALL agents' thoughts
              ↓
      Emergent insight: blends
      research + criticism +
      vision + pragmatism in
      ways no agent intended
```

## Why this can't be done with a shared database

| Shared database | Shared SDM |
|---|---|
| Agents read each other's exact outputs | Agents' thoughts interfere and blend |
| Retrieving = looking up what was stored | Reconstructing = creating something new from superposition |
| 4 agents → 4 separate perspectives | 4 agents → emergent patterns from interference |
| More agents = more data to search | More agents = richer interference = better emergence |

## Quick start

```bash
# Start HeatherDB
cargo run --release -p heather_server -- \
  --data-dir /tmp/swarm_db \
  --dimension 384 \
  --port 6380

# Setup
cd sample_projects/swarm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "HEATHER_URL=http://localhost:6380" >> .env

# Run
python run.py
```

### Commands

| Command | Description |
|---------|-------------|
| `think <question>` | All 4 agents think, write to shared EAM, extract emergent patterns |
| `ask <question>` | Query existing shared memory (no new agent contributions) |
| `agents` | Show the 4 agents and their roles |
| `stats` | HeatherDB stats + thought counts per agent |
| `quit` | Exit |

### Example

```
swarm> think How should we design a Mars colony?

  ○ researcher (Research & Knowledge) thinking... 7 thoughts
  ○ critic (Risks & Challenges) thinking... 6 thoughts
  ○ visionary (Possibilities & Connections) thinking... 7 thoughts
  ○ pragmatist (Implementation & Action) thinking... 6 thoughts

  ✧ extracting emergent patterns from shared memory...
  ✧ synthesizing...

============================================================
  ✧ Emergent Patterns (reconstructed from interference)
============================================================

  [visionary | strong 87%]
  Mars colony architecture could mirror mycorrhizal networks...

  [pragmatist | moderate 74%]
  Initial habitats should use in-situ resource utilization...

  [critic | moderate 71%]
  Radiation exposure during transit alone exceeds lifetime limits...

  [researcher | moderate 68%]
  The psychological effects of isolation mirror Antarctic stations...

============================================================
  ✧ Synthesis
============================================================

  The interference patterns reveal a tension between the visionary's
  biological architecture and the critic's radiation constraints that
  neither agent addressed directly: underground mycorrhizal-inspired
  networks would simultaneously solve radiation shielding AND create
  the organic connectivity the pragmatist's ISRU approach needs...
```

### Cross-topic queries

After thinking about multiple topics, use `ask` to find emergent connections:

```
swarm> think How should we design a Mars colony?
swarm> think What can we learn from ant colonies?
swarm> think How do cities evolve organically?

swarm> ask What connects biological systems and human habitats?
  (SDM reconstructs from the interference of ALL previous thoughts)
```

## The 4 agents

| Agent | Role | What it contributes |
|-------|------|---------------------|
| **Researcher** | Research & Knowledge | Facts, data, scientific insights |
| **Critic** | Risks & Challenges | Failure modes, counterarguments, hidden costs |
| **Visionary** | Possibilities & Connections | Cross-domain analogies, creative possibilities |
| **Pragmatist** | Implementation & Action | Concrete steps, resources, trade-offs |

## Project structure

```
swarm/
  .env                  # ANTHROPIC_API_KEY, HEATHER_URL
  requirements.txt      # Dependencies
  swarm/
    __init__.py
    client.py           # HeatherDB HTTP client
    embeddings.py       # sentence-transformers + SQLite cache (with agent tags)
    agents.py           # 4 specialized thinking agents
    swarm.py            # Orchestrator: agents → SDM → emergence → synthesis
    cli.py              # Interactive terminal
  run.py                # Entry point
```

## The AGI argument

Current AI systems are single minds with context windows. Multi-agent frameworks are groups of single minds passing notes. Neither resembles how biological intelligence works — vast networks of neurons sharing a substrate, with intelligence emerging from interference patterns.

Swarm is a proof of concept for a different paradigm: **intelligence through shared associative memory**. The more agents contribute, the richer the interference patterns, the more surprising the emergent insights. This scales in ways that message-passing can't — you're not adding more voices to a conversation, you're adding more patterns to a superposition.
