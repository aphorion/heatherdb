# Muse — Creative Idea Blender

A creative tool that blends unrelated concepts into surprising ideas, powered by HeatherDB's Elastic Associative Memory. Demonstrates EAM's **blending property** — the key capability that RAG and vector search can't replicate.

## Why this can't be done with RAG

When you query a vector database with the average of "jazz" and "gothic architecture", it returns whichever stored chunk happens to be nearest — one or the other, not a blend.

SDM is different. It stores vectors **superimposed** into shared memory locations. When you query with a blended vector, the Hopfield dynamics reconstruct a coherent pattern from the interference of everything stored — a genuine blend that pulls toward multiple concepts simultaneously.

## How it works

```
Concept A ("jazz")  ──→  embed  ──→  vec_a  ──┐
                                               ├──→  average  ──→  query HeatherDB
Concept B ("architecture")  ──→  embed  ──→ vec_b  ──┘              ↓
                                                              SDM reconstructs
                                                              a blended pattern
                                                                    ↓
                                                           find nearby concepts
                                                                    ↓
                                                         Claude generates idea
```

## Quick start

### Prerequisites

- HeatherDB server (from this repo)
- Python 3.10+
- Anthropic API key

### Setup

```bash
# Start HeatherDB (use a separate data dir from other projects)
cargo run --release -p heather_server -- \
  --data-dir /tmp/muse_db \
  --dimension 384 \
  --port 6380

# In another terminal
cd sample_projects/muse
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "HEATHER_URL=http://localhost:6380" >> .env

# Run
python run.py
```

On first run, Muse pre-seeds 10 diverse concepts (jazz, architecture, deep sea creatures, fractals, tea ceremonies, quantum physics, street art, mycorrhizal networks, origami, aurora borealis).

### Commands

| Command | Description |
|---------|-------------|
| `add <concept>` | Store a new concept (word, phrase, or sentence) |
| `blend <a> + <b> [+ ...]` | Blend 2+ concepts into a creative idea |
| `spark <concept>` | Free-associate: heavy noise on one concept |
| `explore` | Random walk through concept space |
| `concepts` | List all stored concepts |
| `stats` | HeatherDB statistics |
| `quit` | Exit |

### Example session

```
muse> blend jazz + gothic architecture
  inputs: jazz, gothic architecture
  associations:
    [moderate 72%] origami and mathematical paper folding
    [moderate 68%] jazz improvisation and syncopated rhythms
    [faint 55%] gothic cathedral architecture with flying buttresses

  "Resonant Vaults" — A cathedral whose ribbed vaults are tuned like
  a marimba. Each flying buttress is a different pitch; rain becomes
  percussion, wind becomes melody. The building improvises with weather.

muse> spark bioluminescent deep sea creatures
  inputs: bioluminescent deep sea creatures
  associations:
    [strong 81%] the aurora borealis and solar winds
    [moderate 65%] quantum entanglement and superposition
    ...

  "Quantum Lanterns" — Deep-sea organisms whose bioluminescence is
  triggered by quantum tunneling events, creating light patterns that
  mirror the aurora above the ocean surface...

muse> explore
  1. [78%] fractal geometry in nature
  2. [71%] mycorrhizal networks connecting forest trees
  3. [64%] ancient Japanese tea ceremony rituals
  4. [59%] origami and mathematical paper folding
  5. [52%] quantum entanglement and superposition
```

## Project structure

```
muse/
  .env                  # ANTHROPIC_API_KEY, HEATHER_URL
  requirements.txt      # Dependencies
  muse/
    __init__.py
    client.py           # HeatherDB HTTP client
    embeddings.py       # sentence-transformers + SQLite cache
    blender.py          # Core: blending, sparking, exploration
    cli.py              # Interactive terminal
  run.py                # Entry point with concept pre-seeding
```
