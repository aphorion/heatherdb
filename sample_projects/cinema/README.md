# Cinema — Movie Recommendations from Emergent Memory

Cinema is a movie recommendation engine that builds user taste profiles using **HeatherDB's Elastic Associative Memory (SDM)** instead of traditional collaborative filtering or matrix factorization. Every movie is encoded as a 128-dimensional vector, and user preferences emerge naturally from the superposition of watched movie vectors in SDM.

## How It Works

### 1. Movie Encoding

Each movie is encoded into a **128-dimensional vector** using deterministic random projections. The encoding combines four weighted components:

| Component | Weight | Source |
|-----------|--------|--------|
| Genres | 35% | Genre labels (Action, Sci-Fi, etc.) |
| Plot keywords | 30% | Top 10 content words from the plot summary |
| Director | 20% | Director name |
| Actors | 15% | Top 4 billed actors |

Each feature (e.g. `genre:sci-fi`, `director:christopher nolan`) is hashed via SHA-256 to seed a deterministic random unit vector. The weighted sum of all feature vectors produces the final movie embedding. No neural network or training is involved — the vectors are purely deterministic from metadata.

### 2. Taste Fingerprints via EAM

When a user watches a movie, its 128d embedding is **written to the user's SDM collection** in HeatherDB. Over time, the collection accumulates a superposition of all watched movie vectors across the EAM's hard locations.

The **taste fingerprint** is computed entirely server-side by HeatherDB via `GET /collections/{name}/fingerprint`:

1. **Weighted centroid**: Compute the write-count-weighted average of all hard location patterns — locations that absorbed more writes (popular taste regions) contribute more
2. **Hopfield refinement**: Feed the centroid through EAM's iterative Hopfield read to sharpen it into the nearest attractor in memory space

The result is a single vector that captures genre preferences, directorial affinities, and thematic patterns — without any client-side computation or explicit preference modeling. The EAM *is* the user's taste.

#### Evaluation vs. Client-Side Centroid

The server-side fingerprint was benchmarked against the old method (compute centroid in Python → read from EAM):

| Metric | Old (Python) | New (HeatherDB) |
|--------|-------------|-----------------|
| Latency | 2.3ms | 0.7ms (**3.3x faster**) |
| Rank correlation | — | Spearman ρ = 1.0 (identical rankings) |
| Leave-one-out mean rank | 82.2 | **49.2** (lower = better) |
| Hit@20 rate | 20% | **40%** |
| Outlier drift | 0.0001 | 0.0008 (both negligible) |

The new method produces identical or better recommendations with zero client state.

### 3. Recommendations

Movies are ranked by **cosine similarity** between the user's taste fingerprint and each catalog movie's embedding. The result is a ranked list where:
- High similarity = closely aligned with the user's established taste
- Low similarity = far from their comfort zone (potential discovery)

### 4. Fidelity Metrics

Cinema tracks two fidelity scores that measure how well SDM is learning:

- **Fingerprint precision**: Cosine similarity between the centroid query and EAM's reconstruction. High values mean the memory accurately represents the user's taste.
- **Memory recall**: Average cosine similarity when reading back each individual movie embedding. High values mean EAM can faithfully reconstruct specific memories.

These metrics improve as more movies are watched — EAM's adaptive hard locations specialize to the user's taste region.

### 5. "Why This?" — Activation Tracing

Cinema can explain *why* a movie was recommended by tracing EAM's internal activation pattern. For a recommended movie:

1. **Analyze the recommendation**: Query the movie's embedding against the user's SDM collection via `/analyze`, which returns which hard locations activated and their Hopfield-weighted contributions
2. **Analyze each watched movie**: Same call for every movie in the user's history, building a map of which locations each movie "owns"
3. **Compute overlap**: For each watched movie, multiply the shared location weights — locations activated by both the recommendation and the watched movie indicate a memory-level connection
4. **Normalize**: Express each movie's contribution as a percentage

The result: "Recommended because of Interstellar (42%), Blade Runner (31%), Arrival (27%)". This is not cosine similarity between embeddings — it reflects how SDM actually stored and retrieves patterns through its hard location topology. Two movies with moderate embedding similarity can have high explanation weight if competitive learning placed them in the same memory regions.

### 6. Taste Evolution

Cinema tracks how the user's Hopfield fingerprint drifts over time. After each watch, a snapshot records:

- **Fingerprint stability**: Cosine similarity between the previous and new fingerprint — early watches cause large shifts (unstable basin), later watches cause small shifts (settled attractor)
- **Hopfield iterations**: How many iterations the read took to converge — more iterations suggest the memory is still forming its attractor landscape
- **Top genres**: The current genre affinity profile

The timeline visualizes attractor basin stabilization: the stability score monotonically increases as EAM's hard locations specialize to the user's taste region.

### 7. Taste Blending

Two users' fingerprints can be **blended** by averaging their SDM-reconstructed vectors. The blended vector is then used to rank the catalog, producing recommendations that satisfy both users' tastes. A **taste compatibility score** (cosine similarity between the two fingerprints) indicates how similar their preferences are.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (React + Vite + Tailwind)             │
│  Browse / For You / My List / Blend / Lab       │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (port 5173 → proxy → 8000)
┌──────────────────▼──────────────────────────────┐
│  Backend (FastAPI)                               │
│  - OMDb proxy (movie search & metadata)          │
│  - Encoder (movie → 128d vector)                 │
│  - Engine (catalog, users, recommendations)      │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (port 6380)
┌──────────────────▼──────────────────────────────┐
│  HeatherDB                                       │
│  - Per-user SDM collections                      │
│  - Adaptive hard locations (1000–2000)           │
│  - k=20 nearest neighbor activation              │
│  - Hopfield iterative read (beta=5.0)            │
└─────────────────────────────────────────────────┘
```

## Running

**1. Start HeatherDB:**
```bash
cargo run --release -- --data-dir ./data --dimension 128 --port 6380
```

**2. Start the backend:**
```bash
cd sample_projects/cinema/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**3. Start the frontend:**
```bash
cd sample_projects/cinema/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Create a profile, click "Load Popular Films" to seed ~200 curated movies, then start watching to build your taste fingerprint.

## Configuration

Environment variables in `sample_projects/cinema/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OMDB_API_KEY` | `7fc8104c` | OMDb API key for movie metadata |
| `HEATHER_URL` | `http://localhost:6380` | HeatherDB server URL |
| `HEATHER_DIM` | `128` | Vector dimension |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/search?q=...` | Search OMDb for movies |
| `GET` | `/api/movie/{imdb_id}` | Get movie details from OMDb |
| `POST` | `/api/catalog/add` | Add movie to catalog (fetches from OMDb + encodes) |
| `GET` | `/api/catalog` | List all catalog movies |
| `POST` | `/api/seed` | Seed catalog with ~200 curated popular films |
| `GET` | `/api/users` | List all user profiles |
| `POST` | `/api/users/{name}/watch` | User watches a movie (writes to SDM) |
| `GET` | `/api/users/{name}/history` | Get user's watch history |
| `GET` | `/api/users/{name}/fingerprint` | Get user's taste fingerprint + genre affinities |
| `GET` | `/api/users/{name}/recommend` | Get ranked recommendations |
| `GET` | `/api/users/{name}/recommend/{imdb_id}/explain` | Explain why a movie was recommended (activation tracing) |
| `GET` | `/api/users/{name}/evolution` | Get taste evolution timeline |
| `DELETE` | `/api/users/{name}` | Delete a user profile |
| `POST` | `/api/blend` | Blend two users' fingerprints |
| `POST` | `/api/demo/stability` | Demo: fingerprint stability under outliers |
| `POST` | `/api/demo/coldstart` | Demo: cold start progression |

## What Makes This Different

Traditional recommendation systems use collaborative filtering ("users who liked X also liked Y") or content-based filtering with trained models. Cinema uses **neither**:

- **No training**: Movie vectors are deterministic from metadata — no gradient descent, no epochs, no loss functions
- **No user-item matrix**: SDM stores distributed patterns, not explicit ratings
- **Emergent preferences**: Taste fingerprints emerge from the superposition of movie vectors in EAM's address space, not from explicit preference modeling
- **One-shot learning**: A single watch immediately influences the taste fingerprint — no need for batch retraining
- **Graceful degradation**: EAM's distributed storage means outlier watches (one romance in a sea of sci-fi) barely shift the fingerprint, providing natural noise resistance

The key insight is that EAM's properties — distributed storage, superposition, and associative recall — naturally produce the kind of soft clustering and generalization that recommendation systems typically require neural networks to achieve.

## What the Server-Side Fingerprint Unlocks

The fingerprint turns each HeatherDB collection into a **self-summarizing knowledge store** — it knows what it contains without enumeration. This opens up capabilities beyond basic recommendations:

- **Stateless clients**: No need to track user history or compute centroids client-side. Call `GET /fingerprint` and you have the user's complete taste in one vector.
- **Cross-application identity**: A single collection can hold movie vectors, book vectors, and music vectors. The fingerprint fuses them all — one portable taste profile across apps.
- **Real-time drift detection**: Compare fingerprints over time. A sudden drop in `cosine(fp_t0, fp_t1)` signals a taste shift — regime-change detection for free.
- **Collection arithmetic**: `fingerprint(alice) + fingerprint(bob)` = blended taste. `fingerprint(scifi_fans) - fingerprint(comedy_fans)` = what separates them. Cluster users by fingerprint similarity for automatic cohorts.
- **Cold-start transfer**: New user watches 2 movies? Find the nearest existing fingerprint and bootstrap recommendations from that neighbor. One API call, no training.
- **Forgetting / decay**: A merge or decay operation on hard locations would let old preferences fade naturally — temporal taste without timestamps.
- **Novelty scoring**: Compare a candidate movie's embedding against the user fingerprint. High divergence = surprising recommendation, low = safe pick — the threshold slider computed server-side.
- **Multi-modal fusion**: Write image embeddings, text embeddings, audio features to the same collection. The fingerprint fuses them in superposition — one vector summarizing everything the memory has absorbed.
