# Asking what isn't there

## Where you were

Three lessons in, every question you have asked has had an answer worth
having. You wrote patterns and read them back; you built records and pulled
their fields out. The engine cooperated because you only ever asked it about
things it knew.

Fidelity has been in the background the whole time — the cosine between what
you asked and what came back, computable after every read. So far you have used it
to check the engine's work. It is also information about the *question*.

## What this lesson shows

This lesson turns that gap into a decision. You will store only normal
behaviour — no anomalies, nothing labelled — and then use fidelity as the
detector. The same read path you have used all along, asked to make a
judgement rather than fetch an answer.

The thing to watch for: **nothing here is trained to detect anything.** There
is no anomaly model, no labelled examples of bad data, and no second system.

## 1. Store what normal looks like

```bash
curl -s -u admin:tutorial-password \
  -H 'Content-Type: application/json' \
  -d '{"name": "watch", "dimension": 128}' \
  http://localhost:6380/db
```

Save as `lesson4.py`:

```python
from heather import call, unit, cos, jitter
import random, math

random.seed(19)
D = 128

def symbol():
    return unit([random.gauss(0, 1) for _ in range(D)])

# Three normal regimes — think "weekday traffic", "weekend traffic",
# "overnight batch". Real deployments have more; the shape of the lesson is
# the same.
normal = [symbol() for _ in range(3)]

C = "/db/watch/collections/traffic"

def write(vectors):
    for i in range(0, len(vectors), 100):
        call("POST", C + "/write", {"vectors": vectors[i:i + 100]})

# 600 observations of ordinary behaviour. Note what is absent: no labels, no
# anomalies to learn from, no train/test split. This is just your normal data,
# written the way you would have written it anyway.
write([jitter(normal[i % 3], 0.3) for i in range(600)])

def fidelity(v):
    """Read a vector back and compare it with what the memory returned.
    High means the memory recognises this. Low means it does not."""
    r = call("POST", C + "/read", {"query": v})["result"]
    return cos(v, r)
```

## 2. Ask it about things it has seen, and things it has not

```python
# Twenty fresh observations from the same three regimes — never written,
# but the same kind of thing.
held_out = [jitter(normal[i % 3], 0.3) for i in range(20)]

# Twenty vectors from nowhere. Not corrupted normals — unrelated directions.
anomalies = [symbol() for _ in range(20)]

n = sorted(fidelity(v) for v in held_out)
a = sorted(fidelity(v) for v in anomalies)

print("normal   min %.3f  median %.3f  max %.3f" % (n[0], n[10], n[-1]))
print("anomaly  min %.3f  median %.3f  max %.3f" % (a[0], a[10], a[-1]))
```

```
normal   min 0.944  median 0.956  max 0.966
anomaly  min 0.001  median 0.115  max 0.179
```

The two populations do not overlap, and they are not close to overlapping.
The worst normal scored 0.944; the best anomaly scored 0.179.

## 3. Turn the gap into a rule

With a gap this wide the threshold is not a delicate choice — anywhere in the
empty band works, and halfway between the extremes is honest and simple.

```python
threshold = (n[0] + a[-1]) / 2

print("threshold %.3f -> misses %d of 20 normals, catches %d of 20 anomalies"
      % (threshold,
         sum(1 for x in n if x < threshold),
         sum(1 for x in a if x < threshold)))
```

```
threshold 0.561 -> misses 0 of 20 normals, catches 20 of 20 anomalies
```

That is a working detector, derived from data you already had, applied to a
number the read already gives you.

## 4. When there is no empty band

Real data rarely separates like that. Run the same procedure against the card
transactions from [Encode something real](encode-something-real.md), where the
anomalies are ordinary transactions with one field pushed out of range, and the
band disappears:

```
held-out normals: min 0.754  5th percentile 0.896  median 0.983
```

The lowest legitimate transaction scores 0.754. Some anomalies score higher
than that. There is no line with no cost, so the threshold stops being a
question about the data and becomes a question about what you can afford.

Ask it that way and it is answerable. Pick a false-alarm budget, then take the
threshold from the held-out normals at that percentile — the budget *is* the
percentile — and verify on a second held-out set that was never involved in
choosing it.

```python
# 100 normals to calibrate on, 100 more to check the calibration was honest,
# and neither set was ever written to the memory.
calibration = sorted(fidelity(t) for t in held_out_normals)

def threshold_for(budget_percent):
    """The score below which `budget_percent` of legitimate items already fall.
    Choosing the budget chooses the threshold; there is nothing left to eyeball."""
    return calibration[int(budget_percent / 100 * len(calibration))]
```

| Budget | Threshold | Anomalies caught | False alarms (measured) |
|---|---|---|---|
| 0% | 0.754 | 58% | 0% |
| 5% | 0.896 | 68% | 8% |
| 10% | 0.930 | 70% | 11% |

Read the table as a menu rather than a search for the best row. Tolerating no
false alarms costs ten points of detection. Buying those ten points costs eight
percent of legitimate transactions flagged. Which trade is right is not a
property of the memory — it depends on whether a false alarm means a declined
card or a line in a report nobody reads.

Two properties are worth keeping.

**The budget you request is not the rate you get.** A 5% percentile produced
8% false alarms on the second held-out set, because 100 calibration items only
locate a percentile so precisely. Always measure the realised rate on data that
took no part in the calibration.

**A threshold is valid at the load it was calibrated at.** As a pool grows,
more items compete to answer any query and scores shift. Recalibrate when the
data volume changes materially, and treat a threshold copied from
documentation — including this page — as a guess.

## 4. Degrees of strange

Anomalies are rarely so tidy. More often something familiar goes gradually
wrong, and you want to know how wrong.

```python
# The same regime, progressively more distorted. spread 0.3 is what normal
# data looks like; 1.2 is badly deformed but still built from pattern 0.
for spread in (0.3, 0.6, 0.9, 1.2):
    print("  spread %.1f  fidelity %.3f" % (spread, fidelity(jitter(normal[0], spread))))
```

```
  spread 0.3  fidelity 0.955
  spread 0.6  fidelity 0.848
  spread 0.9  fidelity 0.723
  spread 1.2  fidelity 0.675
```

Fidelity does not collapse — it slides. A thing half-way to strange scores
half-way down. That is what makes it usable as a signal to alert on, chart
over time, or route on, rather than only a yes/no.

## What just happened

You built an anomaly detector without training an anomaly detector. Count what
was *not* required: no labelled anomalies, no autoencoder to fit, no one-class
model, no separate scoring service, no retraining as normal drifts — new
normal behaviour is absorbed by writing it, which is lesson 2 doing its job in
the background.

The reframe worth taking away is this. Every read already tells you how much
the memory recognised the question. Treat that number as an answer in its own
right and a whole category of jobs collapses into the read path:

- **Anomaly detection** — fidelity below a threshold.
- **Cold start** — fidelity on a new user's first vector, telling you whether
  you have enough to personalise yet.
- **Regime change** — the same query's fidelity drifting down over weeks.
- **Abstention** — the system declining to answer rather than guessing, which
  is the difference between a demo and something you can deploy.

One caution. The clean separation above comes from anomalies that are genuinely
unrelated directions. Real anomalies are often *near* normal, the gap narrows,
and the threshold becomes a real decision with a real false-positive cost.
Calibrate it on your own data as above, and see [Calibrate a
gate](../how-to/calibrate-a-gate.md) for the parameter-free option the engine
provides.

## Where to go next

Every number you have read so far went unchecked. Before trusting any of them:
[Know it worked](know-it-worked.md).
