# Encode text

Turn a string — a title, a description, a message, a document — into a vector
whose direction carries what the string is about.

Three routes, with different costs. Pick by what you can afford to depend on.

| Route | Dependency | Meaning comes from |
|---|---|---|
| Embedding model | a model at build time | what the model was trained on |
| Hashed features | none | surface form: words and character shape |
| Self-formed | none | your own corpus, one pass, no training |

## Route 1: an embedding model

The shortest path, when a build-time dependency is acceptable.
`all-MiniLM-L6-v2` is 384-dimensional, about 90 MB, and runs on CPU:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
# normalize_embeddings=True returns unit vectors, which is what a cosine
# store wants; skip it and every downstream norm has to be recomputed.
vec = model.encode(text, normalize_embeddings=True)
```

Two things follow from that width and that training:

- 384 is the model's native width, not your collection's. Coming down is a
  fixed-seed projection — see [Choose a dimension](choose-a-dimension.md).
- Model output is anisotropic and **must** be centred before comparison —
  see [Center your vectors](center-your-vectors.md).

The model is part of your data contract: a version bump re-encodes the corpus,
because vectors from two model versions are not comparable.

## Route 2: hashed features, no dependency

Feature hashing gives a fixed-width vector from any string with nothing but
the standard library. Hash each feature to a component index *and* a sign bit;
the sign is what keeps unrelated features from piling up in the same
direction when they collide.

```python
import hashlib
import re

DIM = 128
TOKEN_RE = re.compile(r"[a-z0-9]+")

def _place(feature: str, vec: list[float], weight: float) -> None:
    """Add one feature into the vector at a hashed index, with a hashed sign."""
    h = hashlib.sha256(feature.encode()).digest()
    idx = int.from_bytes(h[:4], "little") % DIM
    # Bit 0 of a later byte gives an unbiased +/-1. Two different features
    # landing on the same index then cancel on average instead of summing,
    # so a collision costs noise rather than a false match.
    sign = 1.0 if h[4] & 1 else -1.0
    vec[idx] += sign * weight

def encode(text: str, word_w: float = 1.0, tri_w: float = 0.5) -> list[float]:
    vec = [0.0] * DIM
    tokens = TOKEN_RE.findall(text.lower())

    # Words carry topic. They are also brittle: "authenticate" and
    # "authentication" are unrelated features to a word hasher.
    for t in tokens:
        _place(f"w:{t}", vec, word_w)

    # Character trigrams repair that brittleness -- the two words above share
    # most of their trigrams -- and survive typos, inflection and unseen
    # vocabulary. They are weaker evidence than a whole word, hence tri_w<1.
    for t in tokens:
        padded = f" {t} "
        for i in range(len(padded) - 2):
            _place(f"c:{padded[i:i+3]}", vec, tri_w)

    n = sum(x * x for x in vec) ** 0.5
    return [x / n for x in vec] if n else vec
```

This is a lexical encoder: it matches strings that *look* alike, and does not
know "physician" and "doctor" are related. Route 3 does.

## Route 3: self-formed co-occurrence

Start every word as a random unit vector — near-orthogonal to every other
word, meaning nothing. Then write co-occurrence pairs into a collection. The
memory's superposition does the rest, in one pass, with no gradient descent
anywhere.

```python
STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for"}

def word_vec(word: str) -> np.ndarray:
    """A word's name tag. Random, fixed, and carrying no meaning at all."""
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)

def ingest(sentence: str, window: int = 5) -> None:
    words = [t for t in TOKEN_RE.findall(sentence.lower())
             if t not in STOP and len(t) > 2]
    # Dedupe while keeping order: a word repeated in one sentence should not
    # get extra pulls on its own neighbours.
    words = list(dict.fromkeys(words))

    batch = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + window, len(words))):
            # The written vector is the *average of a pair*, not either word.
            # Every write is a claim that these two belong near each other;
            # the shared component of frequent pairs reinforces, the rest
            # averages away.
            pair = word_vec(words[i]) + word_vec(words[j])
            batch.append((pair / np.linalg.norm(pair)).tolist())
    call("POST", "/db/lexicon/collections/words/write", {"vectors": batch})
```

Read a word's vector back and you get its blended context, not its name tag —
so compare two reconstructions to compare two words:

```python
def reconstruct(word: str) -> np.ndarray:
    r = call("POST", "/db/lexicon/collections/words/read",
             {"query": word_vec(word).tolist(), "strategy": "iterative"})
    return np.array(r["result"])
```

After one pass over a co-occurrence corpus, `king` and `queen` move from a raw
random-vector similarity of ~0.03 to strongly similar reconstructions, while
`king` and `doctor` — different semantic domain — stay measurably lower. The
underlying name tags never changed; the relation lives in the memory.

Use this when the vocabulary is yours — SKUs, symptoms, ticket tags, log
templates — and no pretrained model has seen it.

## Pool by content, never by count

A document vector is a pooled word vector, and how you pool decides what the
document is about. Raw summation makes every document about its function words.

```python
def pool(tokens: list[str], idf: dict[str, float]) -> np.ndarray:
    """Weight each word by how much it distinguishes documents."""
    v = np.zeros(DIM)
    for t in tokens:
        # IDF: a word appearing in every document has near-zero weight, a
        # rare word has high weight. Count-weighting instead would let "the"
        # outvote every content word in the corpus.
        v += idf.get(t, 1.0) * word_vec(t)
    return v / (np.linalg.norm(v) + 1e-9)
```

With no document-frequency table available, weight each term by its own
`norm²` — the squared length of its unpooled contribution — which approximates
the same thing. Either way, weight by content.

Pooled text vectors need centring; independently drawn word name tags do not.
[Center your vectors](center-your-vectors.md) says which is which.

## Related

- [Encode tabular data](encode-tabular-data.md) — text as one column among
  several.
- [Center your vectors](center-your-vectors.md) — mandatory for pooled output.
- [Choose a dimension](choose-a-dimension.md) — coming down from 384.
- [What similarity means](../explanation/what-similarity-means.md) — the
  standing rules behind all three routes.
