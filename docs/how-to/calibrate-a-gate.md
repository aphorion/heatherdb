# Calibrate a gate

A gate is the threshold that decides whether an answer is worth returning.
Derive it from a labelled held-out sample; do not choose it.

## Why not choose it

A threshold picked by inspecting a few scores is a guess about a distribution
you have seen two points of. On one abstention task, three thresholds chosen
by judgement scored **0%, 12% and 12%**; the same task with a ratio gate whose
cut was derived from held-out data scored **75%**. The procedure below is the
difference between those numbers.

## Procedure

1. **Hold out a labelled sample.** A few dozen queries you can label
   *applies* / *does not apply* is enough. It must not overlap what you wrote
   into the collection you are gating.
2. **Score every held-out item** with the statistic you intend to gate on, and
   keep the two label groups separate.
3. **Put the threshold in the gap** between the groups — the midpoint of the
   empty band, not the mean of everything.
4. **Report coverage and error rate at that threshold**, never a single
   accuracy number. A gate trades one against the other; one number hides the
   trade.
5. **Recalibrate when the pool grows.** A gate is valid only at the load it
   was calibrated at.

## Choose a statistic that carries signal

Two candidates come free with every read. They are not equivalent.

| Statistic | How | Signal |
|---|---|---|
| First-contact similarity | cosine(query, `fast` read) — one step, before iteration | strong |
| Post-iteration similarity | cosine(query, `iterative` read) | none |

**First contact carries the signal.** Measured on a labelled sample: mean
0.818 for queries that apply against 0.304 for queries that do not, ROC AUC
**1.0000**, with an empty band across `[0.61, 0.80]`. A threshold of 0.70 —
the middle of that band — gives **70.6% coverage at 0% errors**.

Similarity measured after the Hopfield loop has converged is ~1.0 by
construction: the loop's job is to move the state onto an attractor, so the
final state agrees with itself. It separates nothing.

```python
# The gate statistic. `strategy: "fast"` is a single step, so the returned
# state is still close to what the query landed on rather than to the
# attractor the loop would have dragged it to.
def first_contact(q):
    r = call("POST", "/db/movies/collections/taste/read",
             {"query": q, "strategy": "fast"})["result"]
    return cos(q, r)
```

## Worked example

```python
# Held-out sample: queries the collection should answer, and queries it
# should refuse. Labels come from you, not from the engine.
applies     = [first_contact(q) for q in heldout_applies]
not_applies = [first_contact(q) for q in heldout_not_applies]

# The band between the highest refusal and the lowest acceptance. If it is
# empty, any cut inside it separates the sample perfectly; its width is how
# much room you have before new data crosses over.
lo, hi = max(not_applies), min(applies)
tau = (lo + hi) / 2
print("empty band [%.2f, %.2f]  ->  tau = %.2f" % (lo, hi, tau))

# Report the trade, not an accuracy. Coverage is how much of the workload the
# gate lets through; errors are refusals that got through anyway.
scored = [(s, True) for s in applies] + [(s, False) for s in not_applies]
passed = [ok for s, ok in scored if s >= tau]
print("coverage %.1f%%  errors %.1f%%" % (
      100 * len(passed) / len(scored),
      100 * sum(not ok for ok in passed) / max(len(passed), 1)))
```

```
empty band [0.61, 0.80]  ->  tau = 0.70
coverage 70.6%  errors 0.0%
```

Read that as a contract: three queries in ten are refused, and nothing wrong
gets through. Moving `tau` down buys coverage with errors. There is no setting
that buys both, which is why one number cannot describe a gate.

## A gate is valid only at its calibration load

Scores drift as the pool fills — more locations means more competition for
every query, and a statistic calibrated on a sparse collection reads
differently on a dense one. An attention-entropy gate measured across two
pool sizes:

| Pool | AUC |
|---|---|
| 3 285 items | 0.76 – 0.80 |
| 10 940 items | 0.32 – 0.39 |

At 10 940 the gate is worse than a coin flip, on the same data and the same
code. Recalibrate on a fresh held-out sample whenever the collection grows by
a factor that matters to you, and record the pool size beside the threshold so
a stale gate is visible rather than silent.

## The parameter-free option

If the thing you actually want tuned is the attention temperature rather than
a pass/fail cut, the engine will derive it:

```bash
curl -u admin:pw -X POST \
  http://localhost:6380/db/movies/collections/taste/attention/calibrate \
  -H 'Content-Type: application/json' -d '{}'
```
```json
{"beta": 101.9, "dl": 127.7}
```

That sweeps 81 log-spaced betas across `[0.5, 200]` and keeps the one
minimising leave-one-out value-reconstruction description length. Feed it back
as `scale` on [`attention`](../api/reads.md#post-dbdbcollectionsnameattention).

Held-out evaluation inside the engine goes no further than this. `attention`
takes an `exclude_id` that drops one stored location from the activated set,
which is a single leave-one-out read; the batched and per-row leave-one-out
paths live in the `heather_db` Rust library rather than behind an HTTP route.
Over HTTP, hold your evaluation sample out **before** you write it, as above.

Nothing about `attention/calibrate` is a threshold, but the entropy it exposes can be gated the
same way, and a derived entropy gate is competitive with a hand-tuned one:
**top-1 accuracy 0.941 at 35% coverage**, against **0.90** for a gate tuned by
hand. Less coverage, better answers, no parameter to maintain.

## Related

- [Find your metric's floor](find-your-metrics-floor.md) — what a score reads
  on data with no structure
- [Explain a result](explain-a-result.md) — what to show when the gate passes
- [Read endpoints](../api/reads.md) — `read`, `attention`, `attention/mdl`,
  `attention/calibrate`
- [Fidelity](../terms/fidelity.md) · [Tune a database](tune-a-database.md)
