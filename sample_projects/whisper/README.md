# Whisper — Lossy Semantic Compression

Compress entire documents into HeatherDB's SDM. All sentences superimpose into shared memory — then query from any angle and the reconstruction reflects the **gestalt**, not just the nearest chunk.

## Why this can't be done with a vector database

| | Vector DB (RAG) | Whisper (SDM) |
|---|---|---|
| **Storage** | 847 chunks stored separately | 847 sentences superimposed into shared memory |
| **Query** | Returns top-k nearest chunks | Reconstructs a pattern from ALL sentences' interference |
| **"What's the overall theme?"** | Returns the single most relevant sentence | Reconstruction is influenced by every sentence — captures the gestalt |
| **Frequent themes** | All chunks equally retrievable | Frequent topics reconstruct more vividly (reinforced attractors) |
| **Rare details** | Equally retrievable as frequent themes | Faded — less influence on reconstruction (like human memory) |

The key difference: a vector DB retrieves specific stored text. SDM reconstructs an impression influenced by everything stored simultaneously — like how a human "remembers" a book they read. Not verbatim quotes, but the gist.

## Quick start

```bash
# Start HeatherDB
cargo run --release -p heather_server -- \
  --data-dir /tmp/whisper_db \
  --dimension 384 \
  --port 6380

# Setup
cd sample_projects/whisper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "HEATHER_URL=http://localhost:6380" >> .env
```

### Load a document

```bash
python run.py load sample_data/sparse_distributed_memory.txt
```

### Query from any angle

```bash
python run.py ask "How does memory handle noise?"
python run.py ask "What happens when memory is full?"
python run.py ask "What is the overall theme of this document?"
python run.py ask "How is this related to the brain?"
```

### Interactive mode

```bash
python run.py interactive
```

```
recall> What is the overall theme?
  fidelity: 0.72
  fragments the memory surfaced:
    [vivid 84%] SDM distributes each memory across many locations...
    [clear 71%] The reconstruction process naturally handles noise...
    [clear 68%] Frequently stored patterns create stronger signals...
    [faded 54%] The cerebellum has been proposed as a biological...

  The document is fundamentally about how memory works through
  distribution and reconstruction rather than exact storage — a system
  where patterns are spread across shared locations and recalled through
  convergence, mimicking how biological memory operates.

recall> /load another_document.txt
  absorbed 234 sentences

recall> What connects the two documents?
  (reconstruction draws from BOTH documents' superimposed sentences)
```

### Load multiple documents

Load several documents into the same memory. Queries reconstruct from the interference of ALL documents — finding connections across them that no single document contains.

## Commands

| Command | Description |
|---------|-------------|
| `python run.py load <file>` | Absorb a document into memory |
| `python run.py ask "question"` | Query the compressed memory |
| `python run.py interactive` | Interactive mode |
| `python run.py stats` | Show statistics |

### Interactive commands

| Command | Description |
|---------|-------------|
| `/load <file>` | Absorb another document |
| `/sources` | Show loaded documents |
| `/stats` | Show memory statistics |
| `/quit` | Exit |

## Project structure

```
whisper/
  .env                          # ANTHROPIC_API_KEY, HEATHER_URL
  requirements.txt              # Dependencies
  whisper/
    __init__.py
    client.py                   # HeatherDB HTTP client
    embeddings.py               # sentence-transformers + SQLite cache
    memory.py                   # Core: absorb documents, reconstruct from queries
    cli.py                      # CLI subcommands + interactive mode
  run.py                        # Entry point
  sample_data/
    sparse_distributed_memory.txt   # Sample document about SDM itself
```
