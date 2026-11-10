# Choose a read strategy

`read`, `analyze` and `batch_analyze` take a `strategy`: `"fast"` or
`"iterative"`. They answer different questions. `fast` asks "what does this
query look like?". `iterative` asks "which stored item is this query a
damaged copy of?".

## The two strategies

**`fast`: one step.** The query is compared once against the activated
addresses, and the stored values are blended by those weights. The result
still shows where the query landed, so it is a measurement.

**`iterative`: settle, then read once.** The query is re-weighted against
the activated **addresses** until it stops moving (up to `t_max` passes).
That decides which stored item it belongs to. The values are then read
once, at the settled point. Only the addresses are iterated over, never the
values: a value can differ from its own address (a key→value collection, or
a location that has absorbed shares of other writes), and feeding a value
back in as the next query would move the state toward a different item.

Modern Hopfield theory ([Ramsauer et al., *Hopfield Networks is All You
Need*, ICLR 2021](https://arxiv.org/abs/2008.02217)) shows that one update
retrieves a pattern when the stored patterns are well separated. That is
why `fast` is enough for most lookups. Iteration earns its cost only when
the query starts far from the item it belongs to.

## Which one to use

| You want to | Use | Why |
|---|---|---|
| Look up the item nearest a reasonably clean query | `fast` | One step already lands on the item; iteration adds latency and no accuracy |
| Decide whether to trust or abstain from an answer | `fast` | The first-step weights and fidelity carry the signal; see [abstention](../explanation/abstention.md) |
| Read a key→value collection | either | Both return the value stored under the matched key |
| Recover an item from a heavily corrupted or partial query | `iterative` | Each pass pulls the state closer to the item before the value is read |
| Clean up an unbind or compose result to a stored symbol | `iterative` | See [the cleanup loop](../explanation/the-cleanup-loop.md) |
| Inspect weights with `analyze` for a confidence decision | `fast` | `iterative` reports the weights at the settled point, which look alike for every query |

## The measurement

60 random unit vectors written at d = 256, then each read with Gaussian
noise added to the query. "Noise" is the noise vector's length relative to
the stored item's.

| Noise | `fast` returns the right item | `iterative` returns the right item |
|---|---|---|
| 0 | 30 / 30 | 30 / 30 |
| 1.0 | 26 / 30 | 30 / 30 |
| 1.5 | 14 / 30 | 30 / 30 |
| 2.0 | 2 / 30 | 30 / 30 |
| 3.0 | 0 / 30 | 29 / 30 |

On clean queries the two agree. As the query degrades, `fast` blends toward
its neighbours and `iterative` still settles on the right item.

The confidence picture runs the other way. On the same collection, 30
queries near stored items against 30 never-stored queries:

| Statistic | Stored | Never stored |
|---|---|---|
| `fast` top weight (`analyze`) | 0.79 – 0.83 | 0.06 – 0.10 |
| `iterative` top weight (`analyze`) | 0.81 – 0.85 | 0.85 – 0.88 |
| `iterative` settled similarity (`analyze`) | 0.999 | 0.999 |

`iterative` always settles somewhere, so the weights and similarity it
reports at the settled point are the same for a query the store knows and
one it does not. Gate on `fast`.

## Calling it

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/read \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.2,0.3],"strategy":"fast"}'
```

Omitting `strategy` gives `"iterative"`.

## Related

- [Calibrate a gate](calibrate-a-gate.md): the threshold to put on the
  `fast` signal.
- [Abstention](../explanation/abstention.md): why the first step carries the
  confidence signal.
- [The cleanup loop](../explanation/the-cleanup-loop.md): iteration as
  cleanup between composition levels.
- [Hopfield read](../terms/hopfield-read.md): the update itself.
- [Reads API](../api/reads.md).
