"""Lexicon engine — distributional semantics through SDM superposition.

How it works:
  1. Text is tokenized and stop words removed
  2. A sliding context window (default 5 words) moves over the text
  3. Each window's word vectors are summed and normalized
  4. The context vector is written to SDM

After feeding enough text, the EAM has superimposed thousands of context
windows. Querying with a word vector reconstructs the "average context"
of that word — a holographic blend of everything it co-occurs with.

Words that appear in similar contexts produce similar reconstructions.
Meaning emerges from interference patterns.
"""

from __future__ import annotations

import numpy as np

from .client import HeatherClient
from .vocabulary import Vocabulary, tokenize, STOP_WORDS


class Lexicon:
    """Distributional semantics engine powered by SDM."""

    def __init__(self, heather: HeatherClient, vocab: Vocabulary,
                 window_size: int = 5):
        self.heather = heather
        self.vocab = vocab
        self.window_size = window_size
        self.windows_written = 0
        self.sentences_processed = 0

    def _make_pair_vector(self, word_a: str, word_b: str) -> list[float]:
        """Average two word vectors, normalize. Creates a direct association."""
        va = np.array(self.vocab.get(word_a), dtype=np.float64)
        vb = np.array(self.vocab.get(word_b), dtype=np.float64)
        combined = va + vb
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined.tolist()

    def ingest(self, text: str) -> int:
        """Feed text into the EAM. Returns number of pair vectors written.
        For each sentence, writes the average of every pair of content words.
        This creates clean, direct word-to-word associations."""
        tokens = tokenize(text)
        content_words = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

        if len(content_words) < 2:
            return 0

        # Ensure all words have vectors
        for w in content_words:
            self.vocab.get(w)

        # Write all pairs within a window
        batch = []
        unique_words = list(dict.fromkeys(content_words))  # dedupe, keep order
        for i in range(len(unique_words)):
            for j in range(i + 1, min(i + self.window_size, len(unique_words))):
                vec = self._make_pair_vector(unique_words[i], unique_words[j])
                batch.append(vec)

        if batch:
            chunk_size = 100
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i + chunk_size]
                self.heather.write(chunk)
            self.windows_written += len(batch)

        self.sentences_processed += 1
        return len(batch)

    def ingest_corpus(self, text: str, progress_fn=None) -> int:
        """Feed a large text corpus, sentence by sentence."""
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        total = 0
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if sent:
                n = self.ingest(sent)
                total += n
            if progress_fn and (i + 1) % 50 == 0:
                progress_fn(i + 1, len(sentences), total)
        self.vocab.save()
        return total

    def query(self, word: str) -> list[float]:
        """Query SDM with a word vector. Returns the reconstruction —
        a blend of all contexts where this word (or similar) appeared."""
        vec = self.vocab.get(word)
        return self.heather.read(vec)

    def associates(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find words that co-occur with the given word.
        Queries SDM, then finds nearest vocabulary words to the reconstruction."""
        reconstruction = self.query(word)
        return self.vocab.nearest(reconstruction, top_k=top_k,
                                  exclude={word.lower().strip()})

    def similarity(self, word_a: str, word_b: str) -> float:
        """Semantic similarity: how similar are the EAM reconstructions
        of two words? High = they appear in similar contexts."""
        rec_a = np.array(self.query(word_a), dtype=np.float64)
        rec_b = np.array(self.query(word_b), dtype=np.float64)
        dot = np.dot(rec_a, rec_b)
        na, nb = np.linalg.norm(rec_a), np.linalg.norm(rec_b)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    def analogy(self, a: str, b: str, c: str,
                top_k: int = 5) -> list[tuple[str, float]]:
        """Solve: a is to b as c is to ???
        Computes: reconstruction(b) - reconstruction(a) + reconstruction(c)
        and finds the nearest words."""
        rec_a = np.array(self.query(a), dtype=np.float64)
        rec_b = np.array(self.query(b), dtype=np.float64)
        rec_c = np.array(self.query(c), dtype=np.float64)

        # The relationship vector: what changes from a to b?
        target = rec_b - rec_a + rec_c
        norm = np.linalg.norm(target)
        if norm > 0:
            target = target / norm

        return self.vocab.nearest(
            target.tolist(), top_k=top_k,
            exclude={a.lower(), b.lower(), c.lower()}
        )

    def blend(self, *words: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Blend multiple words: average their reconstructions and find
        what concept lives in the intersection."""
        recs = [np.array(self.query(w), dtype=np.float64) for w in words]
        combined = sum(recs) / len(recs)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        exclude = {w.lower().strip() for w in words}
        return self.vocab.nearest(combined.tolist(), top_k=top_k,
                                  exclude=exclude)

    def fidelity(self, word: str) -> float:
        """How well does the EAM recognize this word? High fidelity =
        strong attractor = frequently seen in many contexts."""
        vec = self.vocab.get(word)
        rec = self.heather.read(vec)
        v = np.array(vec, dtype=np.float64)
        r = np.array(rec, dtype=np.float64)
        dot = np.dot(v, r)
        nv, nr = np.linalg.norm(v), np.linalg.norm(r)
        if nv == 0 or nr == 0:
            return 0.0
        return float(dot / (nv * nr))
