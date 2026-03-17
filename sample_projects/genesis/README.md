# Genesis — Artificial Life in Associative Memory

An artificial life simulator where creatures evolve **inside** a Elastic Associative Memory. The EAM isn't storage — it's the universe. No hand-designed fitness function. The memory's own dynamics create natural selection.

## How it works

Every creature is a 384-dimensional vector living inside HeatherDB. The rules of this universe emerge from how SDM works:

| Biological concept | SDM mechanism |
|---|---|
| **Fitness** | Reconstruction fidelity — query your vector, measure how well it comes back |
| **Reproduction** | Vector averaging + mutation noise (two parents blend into offspring) |
| **Death** | Fitness drops below threshold — the memory can't reconstruct you anymore |
| **Ecological niches** | Basins of attraction — regions where the EAM reconstructs well |
| **Carrying capacity** | SDM capacity limits — too many vectors = interference = fitness drops |
| **Speciation** | Basin splitting — one niche diverges into two as the memory self-organizes |
| **Extinction** | Niche collapse — interference from other species overwhelms reconstruction |
| **Fossils** | Dead creatures can't be removed from EAM — they leave traces that shape the landscape |
| **Reinforcement** | Survivors are re-written each generation, strengthening their patterns |

## The simulation loop

Each generation:
1. **Evaluate** — query HeatherDB with each creature's vector, measure reconstruction cosine similarity
2. **Cull** — creatures below the death threshold are removed (but their SDM traces remain)
3. **Reproduce** — fit creatures are selected via tournament, offspring = parent blend + mutation
4. **Reinforce** — survivors are re-written to SDM, strengthening their patterns
5. **Cluster** — creatures grouped by vector similarity into species

## Quick start

### Prerequisites

- HeatherDB server (from this repo)
- Python 3.10+
- Anthropic API key (for narrator)

### Setup

```bash
# Start HeatherDB with a fresh data directory
cargo run --release -p heather_server -- \
  --data-dir /tmp/genesis_db \
  --dimension 384 \
  --port 6380

# In another terminal
cd sample_projects/genesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "HEATHER_URL=http://localhost:6380" >> .env

# Run
python run.py
```

### Commands

| Command | Description |
|---------|-------------|
| *(enter)* | Advance one generation |
| `step N` | Advance N generations |
| `auto` | Auto-run with narration every 5 generations (Ctrl+C to stop) |
| `narrate` | Claude narrates recent evolutionary history |
| `stats` | Show HeatherDB and simulation statistics |
| `quit` | Exit |

### Example output

```
============================================================
  Generation 14  |  pop: 47  |  +12 born  -8 died  |  3 species
============================================================
  ● Alpha       23 creatures  ███████████████ 0.91  thriving
  ● Beta        18 creatures  ██████████░░░░░ 0.74  stable
  ● Gamma        6 creatures  ████████░░░░░░░ 0.53  stressed

  ⚠  PRESSURE: Alpha approaching carrying capacity

============================================================
  Generation 23  |  pop: 41  |  +9 born  -14 died  |  4 species
============================================================
  ● Alpha       15 creatures  █████████████░░ 0.84  stable
  ● Delta       12 creatures  ████████████░░░ 0.79  stable
  ● Beta        14 creatures  █████████░░░░░░ 0.69  stressed
  ☠  EXTINCTION: Gamma has gone extinct

  ✨ SPECIATION: Delta has emerged

  [narrator]
  A quiet catastrophe unfolds in the memory. Gamma, once six strong,
  has vanished — overwhelmed by the expanding Alpha population whose
  patterns now dominate the reconstruction landscape. But from the
  interference, something new: Delta emerges at the boundary where
  Alpha and Beta once competed, carving a niche from the noise.
```

## Why this is radical

Every other artificial life system has hand-designed rules — explicit fitness functions, predator-prey dynamics, resource models. Genesis has **none of that**. The only rules are:

1. You are a vector
2. The EAM tries to remember you
3. If it can't, you die

Natural selection, speciation, extinction, carrying capacity, niche formation — all of it **emerges** from the mathematics of associative memory. The same Hopfield dynamics that model how neurons store patterns IS the natural selection. It's not a metaphor. It's the same mechanism.

## Project structure

```
genesis/
  .env                  # ANTHROPIC_API_KEY, HEATHER_URL
  requirements.txt      # Dependencies (no sentence-transformers needed!)
  genesis/
    __init__.py
    client.py           # HeatherDB HTTP client
    world.py            # Core simulation: Creature, Species, World
    narrator.py         # Claude narrates evolution
    cli.py              # Interactive terminal with visualizations
  run.py                # Entry point
```
