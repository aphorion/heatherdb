# Find your metric's floor

Before believing a score, measure what it reads on data that has none of the
structure you are claiming. That reading is the floor. Your number means
whatever it means *above* the floor, and nothing below it.

## The failure this prevents

A self-inclusive nearest-centroid separability read — build a centroid per
group from all its members, then score each member by which centroid it is
nearest — scores **0.996 on two piles of Gaussian noise** at D = 8192. There
is no structure in the input at all. Every member pulls its own centroid
toward itself, so every member is nearest to it, and the metric reports near
perfect separation of two groups that differ in nothing.

The same comparison, rebuilt so a member cannot vote for its own answer,
floors at **~0.54** — a coin flip, correctly:

```python
# Held-out signature classification. Signatures are built from the TRAINING
# half only, so nothing scores against a summary it contributed to. The
# score is a margin, not a membership test.
def signature(rows):
    s = sum(np.asarray(rows))           # a group's summed direction
    return s / np.linalg.norm(s)

def heldout_score(A, B, rng):
    a_tr, a_te = split_half(A, rng)     # each group split before anything
    b_tr, b_te = split_half(B, rng)     # is summarised
    sa, sb = signature(a_tr), signature(b_tr)
    # sim(x, A) - sim(x, B): positive means the held-out row leans to its own
    # group. Averaged over both test halves, chance sits at 0.5.
    hits = [cos(x, sa) - cos(x, sb) > 0 for x in a_te]
    hits += [cos(x, sb) - cos(x, sa) > 0 for x in b_te]
    return sum(hits) / len(hits)
```

| Metric | Two piles of noise, D = 8192 |
|---|---|
| Self-inclusive nearest-centroid separability | 0.996 |
| Held-out signature classification | ~0.54 |

A metric that cannot fail has not succeeded. If you cannot construct an input
on which the number comes out bad, the number is measuring the procedure, not
the data.

## Procedure

1. **Name the structure you are claiming.** "Items in group A share a
   direction that items in B do not." One sentence, one claim.
2. **Construct a null that destroys exactly that and preserves everything
   else.** Same dimensionality, same norms, same marginal distributions, same
   sample sizes — only the claimed structure removed.
3. **Run the null enough times to get a distribution**, not one draw. Thirty
   is usable, several hundred is better; the cost is one loop.
4. **Report three numbers side by side**: your score, the null distribution,
   and the dumbest baseline that could plausibly work.

### Building the null

Match the destruction to the claim. The wrong shuffle preserves the thing you
meant to remove and produces a null that is as high as your result.

| The claim | The null |
|---|---|
| These features co-vary within an item | shuffle values **within each row** — keeps every row's marginal, destroys the joint |
| These series are aligned in time | rotate **each series independently** by a random offset — keeps autocorrelation, destroys alignment |
| These labels are predictable from the vectors | **permute the labels** — keeps the geometry entirely, destroys the association |
| These two collections overlap | resample **one** collection from the pooled items |

```python
# The null distribution, not a null value. Each iteration destroys the
# claimed structure a different way round; the spread tells you how far above
# noise your observed score has to sit before it means anything.
null = np.array([heldout_score(A, permute_labels(B, rng), rng)
                 for _ in range(200)])

print("observed %.3f | null %.3f +/- %.3f | null max %.3f"
      % (observed, null.mean(), null.std(), null.max()))
```

```
observed 0.871 | null 0.502 +/- 0.031 | null max 0.588
```

Quote it in that form. `0.871` alone is a claim; `0.871 against a null of
0.50 ± 0.03` is a measurement.

## When the effect is the largest of N, the null must be on the max of N

Searching 500 candidate axes and reporting the best one is a different
experiment from testing one axis. The comparison for the winner is not the
null distribution of a single trial — it is the distribution of the *maximum*
over 500 trials, which sits far higher.

```python
# Wrong: compares the best of 500 against the null for one.
best = max(score(axis) for axis in axes)

# Right: each null iteration repeats the whole search, so the null is drawn
# from the same "best of 500" procedure the observed number came from.
null_max = [max(score(axis, shuffled(data, rng)) for axis in axes)
            for _ in range(200)]
```

Any selection step counts: picking the best hyperparameter, the best
collection, the best of several gates. Whatever you optimised over goes inside
the null loop.

## Report against a dumb baseline too

The null says the effect is not noise. It does not say the machinery earned
its place. Run the cheapest thing that could work on the same split:

- majority class, for a classification score;
- raw cosine to a group mean, where you used a read;
- a random pair of items, where you scored a matched pair.

A result that beats the null and ties the dumb baseline is real and not worth
the engine. State both, and let the reader see the gap.

## Checklist

- [ ] The null preserves everything except the claimed structure.
- [ ] The null is a distribution, run enough times to have a spread.
- [ ] Any selection over N candidates is inside the null loop.
- [ ] Nothing scores against a summary it contributed to.
- [ ] The reported number sits beside the null and a dumb baseline.
- [ ] An input exists on which the metric would come out bad.

## Related

- [Calibrate a gate](calibrate-a-gate.md) — deriving a threshold from held-out
  data
- [Explain a result](explain-a-result.md) — attribution once the number
  survives
- [What similarity means](../explanation/what-similarity-means.md)
- [Cosine similarity](../terms/cosine-similarity.md) · [High-dimensional
  space](../terms/high-dimensional-space.md)
