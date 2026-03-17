# Emergence — Zero-Shot Capability via Memory Composition

Two specialist agents. One codes. One writes. Neither can document code well.
Add their memories together. The composed agent outperforms both on documentation
— having never completed a single task.

## The Thesis

SDM stores experience as superposed counter traces. When two SDMs are added,
counters superpose. In the overlapping region of task space, traces from both
agents reinforce into a novel hybrid capability that neither possessed alone.

## How It Works

1. **Agent A (Coder)** trains on 20 coding tasks → accumulates traces in `EAM_A`
2. **Agent B (Writer)** trains on 20 writing tasks → accumulates traces in `EAM_B`
3. **Composition**: `EAM_C = EAM_A + EAM_B` (one algebra operation via HeatherDB)
4. **Evaluation**: All agents scored on 5 documentation tasks (technical accuracy + communicative quality)

Documentation is the hybrid task — it requires understanding code (coder's domain)
AND communicating clearly (writer's domain). Neither skill alone is sufficient.

## Quick Start

### 1. Start HeatherDB

```bash
cargo run --release -p heather_server -- \
  --data-dir /tmp/emergence_db \
  --dimension 384 \
  --port 6380
```

### 2. Setup

```bash
cd sample_projects/emergence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run

```bash
python run.py
```

### 5. Demo

```
emergence> demo
```

This runs the full pipeline: train both specialists, compose their memories,
evaluate all agents, and display the results.

## Commands

| Command | Description |
|---------|-------------|
| `train` | Train both specialist agents (Coder + Writer) |
| `compose` | `EAM_coder + EAM_writer → EAM_composed` |
| `evaluate` | Score all agents on documentation benchmark |
| `demo` | Full automated run: train → compose → evaluate |
| `blend <α>` | Weighted blend: `α·Coder + (1-α)·Writer` |
| `stats` | Show EAM statistics for all collections |
| `reset` | Drop all collections and clear cache |

## Architecture

```
Task → Embed (all-MiniLM-L6-v2, 384d)
     → Query EAM (query_documents, top-3)
     → Format experiential prior
     → Claude generates with prior context
     → Score (LLM-as-judge: technical accuracy + communicative quality)
```

**Scoring**: Composite = harmonic mean of technical accuracy and communicative quality.
Harmonic mean penalizes imbalance — you can't score high by excelling on one axis alone.

## Dependencies

- **HeatherDB** with algebra endpoints (heather_algebra crate)
- **sentence-transformers** for embedding tasks into 384-dim vectors
- **Anthropic API** (Claude) for agent execution and scoring
- **~80 API calls** for a full demo run (40 training + 20 evaluation + 20 scoring)
