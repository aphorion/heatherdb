# Selective attention

A memory that weighs every feature and every moment equally recalls the wrong
things. Selective attention lets a database learn which features and which
ages predict what it is asked about, from its own write stream, with no
separate training job.

## What it learns

**Feature attention.** A database names the fields it will be asked to fill
in, as *heads*: `{"outcome": [51, 52]}`. A head scores stored rows by
attention-weighted squared distance over the other fields,

```latex
D(q, x) = \sum_{d \notin \text{head}} \alpha_d \, (q_d - x_d)^2
```

and recalls by blending the nearest rows with weights `softmax(−D)`. The
weights `α_d` decide which features matter. Their overall scale is also the
read's temperature, so no separate `beta` is tuned.

**Time attention.** A collection's history is its rows, newest first. The
`next` read sums them with one weight per age bucket (lags 1, 2, 3–4, 5–7,
8–12, …, roughly Fibonacci). The weights decide how fast the past fades.
They are shared by every collection of the database, so a thousand short
per-user histories teach one recency curve.

## How it learns: predict, then store

Every write is a test before it is a memory. Before a row is stored:

1. each head recalls the row from what is already stored, with the head's
   fields blank, and compares the recall with the row's true values;
2. time attention predicts the row from its collection's history;
3. each attention takes one gradient step (Adam) to shrink its error;
4. the row is stored.

The learned state is saved in the write's own transaction. A batch that
fails leaves both the rows and the attention as they were.

Reads use time attention's *consolidated* weights, a running average of
the learned ones. The latest weights carry the noise of the latest step;
the average does not. (Brains separate fast learning from slow
consolidation for the same reason.)

## Is it supervised?

Time attention is self-supervised: the target is the next write. Feature
attention for a head is self-supervised over the record's own fields. The
memory blanks a field it already holds and learns to recall it, the way a
masked language model is trained. Choosing which field gets a head is the
supervision. A head carries the same information a classifier would use;
the difference is that it is learned in the write path, one row at a time,
by the same mechanism for every task.

## What it changes, measured

From the head-to-head benchmark in `benchmarks/h2h/` (public datasets, same
splits and metrics as the incumbents, learned online in one pass):

| Workload | Without | With selective attention | Strongest incumbent |
|---|---|---|---|
| Propensity scoring, UCI bank marketing (ROC-AUC) | 0.867 | 0.931 | 0.935, gradient-boosted trees |
| Recommendations, MovieLens 1M (NDCG@10) | 0.025 | 0.053 | 0.0625, ALS with time-decayed confidence |

Feature attention found call duration first, without being told. Time
attention learned a steep recency curve, from a flat start, by itself.

## Limits

- Plain databases only (`competitive: false`). The rows are the stored
  vectors, and location ids give write order.
- A recall scans every stored row of its collection, exactly and across
  cores, and so does each write's prediction. At 31,647 rows × 53 fields a
  recall takes about 1 ms and learning while writing about 0.3 ms per row.
  Both grow linearly with the collection, so writing a collection grows
  quadratically.
- Time attention is learned from the order of writes, not wall-clock time.
  Write each collection in the order events happened.
- It does not calibrate surprise against local density. That is the
  per-location spread planned as a follow-up, which closes the fraud
  precision gap in the benchmark.

## See also

- [GET /db/{db}/selective-attention](../api/databases.md#get-dbdbselective-attention),
  [recall](../api/reads.md#post-dbdbcollectionsnamerecall),
  [next](../api/reads.md#get-dbdbcollectionsnamenext)
- [`[eam.selective_attention]`](../reference/configuration.md#eamselective_attention)
- [Attention is a read](attention-is-a-read.md): the raw dot-product
  attention read, a different thing from this.
