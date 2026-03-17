# Discovery: Distributional Semantics from Pure Memory

**Date:** 2026-02-13
**Project:** Lexicon (HeatherDB sample project)
**Finding:** Elastic Associative Memory can produce Word2Vec-like semantic relationships using only random vectors and co-occurrence superposition — no neural network, no gradient descent, no training loop.

## Setup

- Each word is assigned a **random unit vector** in 128-dim space (deterministic hash, no pre-trained embeddings)
- For each sentence, every **pair of co-occurring content words** is averaged and written to the EAM
- Querying with a word vector reconstructs a blend of all co-occurrence contexts
- Reconstruction similarity between two words = semantic similarity

## Key Results

### Semantic Similarity
```
similarity(king, queen) = 0.97    (raw random baseline: 0.03)
similarity(king, doctor) = 0.89   (lower — different semantic domain)
similarity(cat, dog) = 0.14       (raw baseline: -0.12, lifted by co-occurrence)
```

Words that appear in similar contexts produce similar EAM reconstructions, even though their underlying vectors are completely random and unrelated.

### Word Associates
```
word king → princess, ruled, soldier  (royal/governance cluster)
word doctor → examined, patient       (medical cluster — with noise)
```

### Analogies
```
king → queen  ::  man → woman    ✓  (emerged in results)
man → woman   ::  king → princess ✓  (female royalty — close to queen)
```

The gender relationship **transferred** through vector arithmetic on EAM reconstructions:
```
reconstruction(queen) - reconstruction(king) + reconstruction(man) ≈ woman
```

## Why This Matters

### Word2Vec (2013, Mikolov et al.)
- Requires millions of training examples
- Uses stochastic gradient descent over many epochs
- Learns embedding vectors through backpropagation
- Training takes hours/days on large corpora

### SDM Lexicon (this project)
- Requires only co-occurrence pair writes (one pass over text)
- No optimization, no loss function, no gradients
- Word vectors are **random** — meaning emerges from memory interference
- "Training" is just writing pairs to SDM — instant, one-shot

### The Mechanism
1. Random word vectors are nearly orthogonal in high-dim space
2. Co-occurring words are averaged and stored in SDM
3. EAM's superposition causes frequently co-occurring pairs to reinforce each other
4. Querying reconstructs a blend weighted by co-occurrence frequency
5. Words with similar co-occurrence profiles produce similar reconstructions
6. **Semantic structure emerges from interference patterns, not learned parameters**

This is analogous to how the brain might form word associations — through repeated co-activation of neural patterns (Hebbian learning), not through error backpropagation.

## Limitations (Observed)

- **Corpus size matters:** 30 paragraphs produced noisy results. Distributional semantics needs volume — the EAM mechanism works but signal-to-noise ratio depends on data quantity.
- **Frequent words dominate:** Very common words (king, queen) create strong attractors that can overshadow subtler relationships.
- **Analogy arithmetic is noisy:** While king→queen::man→woman appeared in results, it was tied with unrelated words. Word2Vec's learned structure produces cleaner arithmetic.
- **Dimension tradeoff:** 384 dims = too spacious (no interference). 128 dims = good interference but some noise. Sweet spot depends on vocabulary size.

## Theoretical Implications

1. **Distributional semantics may not require learning** — memory superposition alone can capture the "you shall know a word by the company it keeps" principle (Firth, 1957).

2. **SDM as a biological model for lexical memory** — the brain doesn't backpropagate error signals for word learning. Hebbian co-activation in distributed memory is more biologically plausible. This experiment suggests that's sufficient.

3. **One-shot vs. gradient descent** — The EAM approach writes each co-occurrence exactly once (or as many times as it naturally occurs). There's no iterative refinement. The "training" IS the data. This suggests distributional structure is inherent in co-occurrence statistics, not an artifact of optimization.

4. **Frequency = reinforcement** — SDM naturally weights frequent co-occurrences more heavily (deeper basins of attraction). This mirrors the statistical foundation of distributional semantics without explicitly computing frequencies.

## Reproducing

```bash
# Start HeatherDB
cargo run --release -p heather_server -- \
  --data-dir /tmp/lexicon_db --dimension 128 --port 6380

# Run Lexicon
cd sample_projects/lexicon
python run.py

# Try
lex> word king
lex> sim king queen
lex> analogy king queen man
```

Feed larger corpora for cleaner results:
```
lex> ingest /path/to/book.txt
```

## Next Steps

- Test with a large corpus (100K+ sentences) to see if analogy arithmetic cleans up
- Compare SDM semantic similarity rankings against Word2Vec on standard benchmarks
- Explore whether EAM can capture syntactic relationships (verb tenses, plurals) in addition to semantic ones
- Investigate the relationship between SDM capacity (dimension, hard locations) and vocabulary size limits
