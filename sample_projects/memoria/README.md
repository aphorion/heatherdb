# Memoria — AI Agent with Associative Memory

A conversational AI that *remembers* like a human — not through exact retrieval, but through associative reconstruction powered by HeatherDB's Elastic Associative Memory.

## What makes this different from RAG?

| | RAG (Vector Search) | Memoria (SDM) |
|---|---|---|
| **Storage** | Each chunk stored separately | Vectors superimposed into shared memory |
| **Retrieval** | Nearest-neighbor lookup — returns exact stored chunks | Reconstruction — rebuilds a pattern from interference of everything stored |
| **Noisy queries** | Degrades — needs good semantic overlap | Pattern completion — partial cues reconstruct full memories |
| **Frequency** | All chunks equally retrievable | Reinforced patterns recalled more vividly |
| **Blending** | Returns top-k discrete results | Queries between topics reconstruct blended signals |
| **Random queries** | Returns arbitrary nearest chunks | Reconstructs coherent patterns from noise (dream mode) |

## How it works

```
User input
    ↓
Embed text → 384-dim vector (sentence-transformers)
    ↓
Query HeatherDB → reconstructed memory vector
    ↓
Find nearest text in local cache (SQLite cosine search)
    ↓
Build prompt with recalled memories + confidence scores
    ↓
Claude → response
    ↓
Store exchange in HeatherDB + SQLite cache
```

Memories are recalled with confidence labels:
- **Vivid memory** (>85%) — frequently reinforced or very close match
- **Clear memory** (>70%) — good reconstruction
- **Vague impression** (>50%) — faded but still present
- Below 50% — too faded, omitted

## Quick start

### Prerequisites

- HeatherDB server (from this repo)
- Python 3.10+
- Anthropic API key

### Setup

```bash
# Start HeatherDB (384 dimensions for sentence-transformers)
cargo run --release -p heather_server -- \
  --data-dir /tmp/memoria_db \
  --dimension 384 \
  --port 6380

# In another terminal
cd sample_projects/memoria
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "HEATHER_URL=http://localhost:6380" >> .env

# Run
python run.py
```

### Chat

```
you> My favorite programming language is Rust
memoria> Nice! Rust is a great choice...

you> I have a cat named Luna
memoria> Luna! That's a lovely name...

you> What's my favorite language?
  recalled memories:
  [vivid memory 100%] User said: My favorite programming language is Rust...
memoria> Your favorite programming language is Rust!
```

### Commands

| Command | Description |
|---------|-------------|
| `/memories` | Show HeatherDB stats and local memory count |
| `/dream` | Dream mode — chain random associations through memory |
| `/quit` | Exit |

## Dream mode

Type `/dream` to watch the AI free-associate through its memories:

1. Picks a random stored vector
2. Adds 10-30% noise
3. Queries HeatherDB — SDM reconstructs a pattern from the noise
4. Finds the nearest text, uses it as the next query
5. Chains 4 hops, each triggering the next by association
6. Claude reflects on the chain

This is impossible with RAG — you can't query a vector database with random noise and get coherent, meaningfully connected results. EAM's basins of attraction make it work.

## Project structure

```
memoria/
  .env                  # ANTHROPIC_API_KEY, HEATHER_URL
  requirements.txt      # Dependencies
  memoria/
    __init__.py
    client.py           # HeatherDB HTTP client
    embeddings.py       # sentence-transformers + SQLite cache
    agent.py            # Memory encode/recall + Claude conversation
    cli.py              # Interactive terminal loop
  run.py                # Entry point
```
