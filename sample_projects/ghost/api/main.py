"""
Ghost — SDM learns your writing patterns in real-time.

Each word pair (bigram) you type gets written to the EAM as a 128-dim vector:
  [context_64 ; target_64]

To predict the next word, query with [context_64 ; zeros_64].
The EAM's reconstruction fills in the target half from superposition
of all words that ever followed that context.

No neural network. No pre-training. The EAM IS the intelligence.
"""

import hashlib
from dataclasses import dataclass, field

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HEATHER_URL = "http://localhost:6380"
CONTEXT_DIMS = 64
TARGET_DIMS = 64
TOTAL_DIMS = CONTEXT_DIMS + TARGET_DIMS  # 128 — HeatherDB must run at this

app = FastAPI(title="Ghost")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── vector hashing (same trick as Lexicon) ──────────────────────────

_vec_cache: dict[str, np.ndarray] = {}


def word_to_vec(word: str) -> np.ndarray:
    """Deterministic random unit vector from word hash."""
    key = word.lower().strip()
    if key in _vec_cache:
        return _vec_cache[key]
    h = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(CONTEXT_DIMS).astype(np.float64)
    vec /= np.linalg.norm(vec)
    _vec_cache[key] = vec
    return vec


# ── state ────────────────────────────────────────────────────────────

@dataclass
class GhostState:
    vocab: dict[str, np.ndarray] = field(default_factory=dict)
    patterns_written: int = 0
    words_seen: int = 0


state = GhostState()


def normalize(word: str) -> str:
    return word.lower().strip().strip(".,!?;:\"'()[]{}—–-…")


# ── heather client ──────────────────────────────────────────────────

async def heather_write(vectors: list[list[float]]):
    async with httpx.AsyncClient() as c:
        await c.post(f"{HEATHER_URL}/write", json={"vectors": vectors}, timeout=10)


async def heather_read(query: list[float]) -> list[float]:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{HEATHER_URL}/read",
            json={"query": query, "strategy": "iterative"},
            timeout=10,
        )
        return r.json()["result"]


# ── core logic ──────────────────────────────────────────────────────

async def write_bigram(w1: str, w2: str):
    """Write [context; target] to SDM."""
    ctx = word_to_vec(w1)
    tgt = word_to_vec(w2)
    combined = np.concatenate([ctx, tgt])
    combined = combined / np.linalg.norm(combined)
    await heather_write([combined.tolist()])
    state.patterns_written += 1


async def write_trigram(w1: str, w2: str, w3: str):
    """Write [context1+context2; target] to SDM for richer context."""
    ctx = word_to_vec(w1) + word_to_vec(w2)
    ctx = ctx / np.linalg.norm(ctx)
    tgt = word_to_vec(w3)
    combined = np.concatenate([ctx, tgt])
    combined = combined / np.linalg.norm(combined)
    await heather_write([combined.tolist()])
    state.patterns_written += 1


async def predict(context: list[str], k: int = 5) -> list[dict]:
    """Predict next word from context using EAM reconstruction."""
    if not context or len(state.vocab) < 2:
        return []

    # try trigram context first, fall back to bigram
    if len(context) >= 2:
        ctx = word_to_vec(context[-2]) + word_to_vec(context[-1])
        ctx = ctx / np.linalg.norm(ctx)
    else:
        ctx = word_to_vec(context[-1])

    query = np.concatenate([ctx, np.zeros(TARGET_DIMS)])
    query = query / np.linalg.norm(query)

    reconstruction = np.array(await heather_read(query.tolist()))

    # extract target half
    target_half = reconstruction[CONTEXT_DIMS:]
    norm = np.linalg.norm(target_half)
    if norm < 1e-8:
        return []
    target_half = target_half / norm

    # find nearest vocab words
    scores: list[tuple[str, float]] = []
    for word, vec in state.vocab.items():
        sim = float(np.dot(target_half, vec))
        if sim > 0:
            scores.append((word, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    # also try bigram-only prediction if we used trigram
    if len(context) >= 2:
        ctx_bi = word_to_vec(context[-1])
        query_bi = np.concatenate([ctx_bi, np.zeros(TARGET_DIMS)])
        query_bi = query_bi / np.linalg.norm(query_bi)
        recon_bi = np.array(await heather_read(query_bi.tolist()))
        tgt_bi = recon_bi[CONTEXT_DIMS:]
        norm_bi = np.linalg.norm(tgt_bi)
        if norm_bi > 1e-8:
            tgt_bi = tgt_bi / norm_bi
            for word, vec in state.vocab.items():
                sim = float(np.dot(tgt_bi, vec))
                if sim > 0:
                    # boost words that appear in both predictions
                    for i, (w, s) in enumerate(scores):
                        if w == word:
                            scores[i] = (w, s + sim * 0.5)
                            break
                    else:
                        scores.append((word, sim * 0.5))
            scores.sort(key=lambda x: x[1], reverse=True)

    # don't predict the context word itself
    context_set = {w.lower().strip() for w in context}
    results = []
    for word, sim in scores:
        if word in context_set:
            continue
        results.append({"word": word, "confidence": round(sim, 4)})
        if len(results) >= k:
            break

    return results


# ── routes ──────────────────────────────────────────────────────────

class TypeRequest(BaseModel):
    word: str
    context: list[str] = []  # previous words for context


class TypeResponse(BaseModel):
    predictions: list[dict]
    stats: dict


@app.post("/api/type", response_model=TypeResponse)
async def type_word(req: TypeRequest):
    """Process a new word: write bigrams and return predictions."""
    w = normalize(req.word)
    if not w:
        return TypeResponse(
            predictions=[],
            stats={
                "vocab_size": len(state.vocab),
                "patterns": state.patterns_written,
                "words": state.words_seen,
            },
        )

    # register in vocab
    if w not in state.vocab:
        state.vocab[w] = word_to_vec(w)
    state.words_seen += 1

    # write bigram with previous word
    ctx = [normalize(c) for c in req.context if normalize(c)]
    if len(ctx) >= 1:
        prev = ctx[-1]
        if prev not in state.vocab:
            state.vocab[prev] = word_to_vec(prev)
        await write_bigram(prev, w)

    # write trigram if enough context
    if len(ctx) >= 2:
        prev2 = ctx[-2]
        if prev2 not in state.vocab:
            state.vocab[prev2] = word_to_vec(prev2)
        await write_trigram(prev2, ctx[-1], w)

    # predict next word
    predictions = await predict(ctx + [w])

    return TypeResponse(
        predictions=predictions,
        stats={
            "vocab_size": len(state.vocab),
            "patterns": state.patterns_written,
            "words": state.words_seen,
        },
    )


class FeedRequest(BaseModel):
    text: str


class FeedResponse(BaseModel):
    words_processed: int
    patterns_written: int
    predictions: list[dict]
    stats: dict


@app.post("/api/feed", response_model=FeedResponse)
async def feed_text(req: FeedRequest):
    """Feed a chunk of text — writes all bigrams and trigrams at once."""
    words = req.text.split()
    cleaned = [normalize(w) for w in words]
    cleaned = [w for w in cleaned if w]

    if not cleaned:
        return FeedResponse(
            words_processed=0,
            patterns_written=0,
            predictions=[],
            stats={
                "vocab_size": len(state.vocab),
                "patterns": state.patterns_written,
                "words": state.words_seen,
            },
        )

    initial_patterns = state.patterns_written

    # register all words
    for w in cleaned:
        if w not in state.vocab:
            state.vocab[w] = word_to_vec(w)
        state.words_seen += 1

    # write all bigrams
    for i in range(len(cleaned) - 1):
        await write_bigram(cleaned[i], cleaned[i + 1])

    # write all trigrams
    for i in range(len(cleaned) - 2):
        await write_trigram(cleaned[i], cleaned[i + 1], cleaned[i + 2])

    new_patterns = state.patterns_written - initial_patterns

    # predict from last context
    predictions = await predict(cleaned[-2:])

    return FeedResponse(
        words_processed=len(cleaned),
        patterns_written=new_patterns,
        predictions=predictions,
        stats={
            "vocab_size": len(state.vocab),
            "patterns": state.patterns_written,
            "words": state.words_seen,
        },
    )


@app.post("/api/reset")
async def reset():
    """Clear all state and HeatherDB."""
    state.vocab.clear()
    state.patterns_written = 0
    state.words_seen = 0
    _vec_cache.clear()
    # HeatherDB doesn't have a clear endpoint — restart it for full reset
    return {"cleared": True}


@app.get("/api/stats")
async def stats():
    return {
        "vocab_size": len(state.vocab),
        "patterns": state.patterns_written,
        "words": state.words_seen,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "dims": TOTAL_DIMS}
