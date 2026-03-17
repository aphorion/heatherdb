# Cortex — A Thinking Notebook Powered by Database-Native Intelligence

Cortex is a web application where **HeatherDB IS the intelligence**. Write notes and observations — the database discovers cross-domain patterns, predicts connections, detects novelty, and surfaces insights with no LLM, no ML model, no training loop.

## Architecture

- **Frontend**: Next.js (React/TypeScript) with Tailwind CSS
- **Backend**: FastAPI (Python) with sentence-transformers for embeddings
- **Database**: HeatherDB (Elastic Associative Memory)

## Quick Start

### 1. Start HeatherDB

```bash
cargo run --release -p heather_server -- --data-dir /tmp/cortex_db --dimension 384 --port 6380
```

### 2. Start the API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

### 3. Start the Frontend

```bash
cd web
npm install
npm run dev
```

### 4. Open http://localhost:3000

## Features

| Panel | What it does |
|-------|-------------|
| **Write** | Add notes to your notebook. Shows familiarity after writing. |
| **Think** | Ask your notebook a question. Side-by-side EAM vs vector search. |
| **Connect** | Blend 2+ concepts to find where ideas intersect. |
| **Surprise** | Test how novel an idea is to your notebook. |
| **Dream** | Free association chain from random noise through the database. |
| **Landscape** | Map the attractor basins — see what your notebook knows most about. |
| **Notes** | Browse, search, and manage all stored notes. |

## How It Works

Every note is embedded using `all-MiniLM-L6-v2` (384 dimensions) and written to HeatherDB's Elastic Associative Memory. When you query:

1. **EAM reconstruction**: The query embedding is read through SDM, which reconstructs a "pattern completion" — what the database collectively remembers about this region of semantic space
2. **Vector search**: Standard k-NN on the raw query embedding
3. **Insights**: Notes that appear in SDM results but NOT in vector search — these are cross-domain connections the database discovered through interference patterns

The EAM doesn't just retrieve — it **thinks**. Overlapping memory traces create emergent associations that pure vector search misses.

## Dimension: 384

Uses `all-MiniLM-L6-v2` sentence-transformer embeddings. At 384 dimensions with ~30 notes, the EAM creates meaningful interference patterns that surface cross-domain connections.
