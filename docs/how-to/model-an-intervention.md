# Model an intervention

Predict what a treatment, perturbation or configuration change will do to a
subject you have never applied it to — from before-and-after measurements
alone, with no equation for the mechanism.

The shape is the same as [Build a world model](build-a-world-model.md): a state
and an action give a next state. What differs is where the leverage is. On a
grid you care about the path; here you care about a single step applied to a
subject nobody has tested, and about knowing when that prediction is worthless.

## The setup

You have measured profiles of several subjects — cell lines, patients,
machines, accounts — and you have applied several interventions, though not
every intervention to every subject. Each experiment gives you a pair: the
profile before, the profile after.

The gap you want to close is the untested combination. Trying it costs a month;
the prediction costs two requests.

## Encode the subject and the intervention separately

Give every subject and every intervention its own vector. The subject's is the
measurement; the intervention's is a label, so a
[hash-seeded symbol](../reference/encodings.md) is enough — the intervention
carries no measurement of its own.

```python
# One vector per subject (the measured profile) and one per intervention (a
# name). Keeping them separate is the point: it is what lets a compound learned
# on one subject be applied to another.
sym = {name: unit(profile_or_symbol(name)) for name in subjects + interventions}
```

## Learn what the intervention does, not what happened

The naive move is to store, per subject, everything done to it. That gives one
vector per subject holding its own experiments, and it cannot answer the
question you care about: unbinding an intervention that was never applied to
*that* subject returns noise, because it was never bundled in.

Learn the intervention instead. For each one, take every subject it *was*
applied to and ask what carried the before into the after:

```python
# "What does this compound do", separated from "what it was done to".
# unbind(after, before) is the transform that carries one into the other;
# bundling those over several subjects keeps what they share and cancels
# what is specific to each.
T = {}
for drug in interventions:
    applied_to = [s for (s, d) in observed if d == drug]
    T[drug] = bundle([unbind(outcome(s, drug), sym[s]) for s in applied_to])
```

This is the [mnemonic](../terms/mnemonic.md) pattern: several instances of one
relationship collapse into a single vector that produces the instance nobody
recorded.

## Predict the untested combination

Apply the transform to a subject it was never derived from:

```python
prediction = bind(sym["hypoxic"], T["compound_c"])
```

With `compound_c` held out of the `hypoxic` experiments entirely — 23 observed
pairs, one gap — the prediction lands on the right answer:

```
predicted vs the true outcome         0.917
best rival (a different compound)     0.044
nearest catalogue entry               hypoxic+compound_c (0.917)
```

The rival column is what makes this a result rather than a coincidence. The
prediction is not merely *near* the family of hypoxic outcomes; it is far from
what every other compound would have produced on the same subject.

## Expect less when the effect depends on the subject

The construction above assumes an intervention does something consistent
wherever it is applied. Real interventions often interact with what they are
applied to, and the interacting part is not learnable from other subjects,
because it is by definition not shared with them.

Running the identical pipeline against a truth where half the effect is an
interaction:

| Effect | Predicted vs true | Best rival |
|---|---|---|
| Consistent across subjects | 0.917 | 0.044 |
| Half consistent, half subject-specific | 0.605 | 0.056 |

The prediction still identifies the right outcome, and it is measurably
weaker — the drop is the interaction term, which no amount of data about *other*
subjects can supply. This is the honest ceiling of the method, and it is
visible in the number rather than hidden behind it.

## Guard the prediction with familiarity

The construction returns a well-formed vector for any subject you hand it,
including one the memory has never seen. Nothing about the output announces
that. So check the subject against the panel before trusting the prediction:

```python
# The subjects you have actually measured. bulk_load rather than write, so each
# one stays its own entry instead of being merged into its neighbours.
call("POST", C + "/bulk_load", {"addresses": panel, "counters": panel})

def familiarity(v):
    """How strongly the panel recognises this subject. Low means the prediction
    that follows is an extrapolation, not an interpolation."""
    r = call("POST", C + "/read", {"query": list(v)})["result"]
    return cos(v, r)
```

```
familiarity, a subject in the panel    1.000
familiarity, a subject never measured  0.047
```

That separation is the whole guard. Predictions for subjects near the panel are
interpolations between things you have measured; predictions for subjects far
from it are extrapolations dressed identically. Set a threshold from held-out
subjects rather than choosing one — [Calibrate a gate](calibrate-a-gate.md) —
and refuse below it.

## What to change for your own data

**Subjects that are measurements, not labels.** Encode them as you would any
profile: see [Encode tabular data](encode-tabular-data.md), and
[centre them](center-your-vectors.md) if they come from a model, because
uncentred profiles of the same tissue all look alike and the differences you
care about live in the deviation from the average.

**Dose, time or intensity.** These are continuous, so they are not separate
symbols — a fractional power gives a graded effect where a label gives a step.
See [Encode quantities and time](encode-quantities-and-time.md).

**A sequence of interventions.** That is a [rollout](../terms/rollout.md), with
a cleanup step between each one, and its accuracy decays with length — see
[Build a world model](build-a-world-model.md).

## Where this stops

**It predicts what the data already implies.** A transform derived from other
subjects carries what those subjects had in common. It cannot produce an effect
that appears nowhere in the observations, and it will not tell you that it
failed to.

**A prediction is a hypothesis, not a result.** The output is a direction in a
space you defined, worth an experiment and not a substitute for one.

**Grade it before believing it.** Hold out whole combinations, predict them, and
report the score beside what a trivial baseline achieves — see
[Find your metric's floor](find-your-metrics-floor.md) and
[Know it worked](../tutorials/know-it-worked.md). A pipeline that returns a
confident answer for every input has not been tested.

## Related

- [Build a world model](build-a-world-model.md) — the general case, and rollouts.
- [World models](../explanation/world-models.md) — what a state can be.
- [Store a procedure](store-a-procedure.md) — the same derive-from-instances move.
- [Calibrate a gate](calibrate-a-gate.md) — where the refusal threshold comes from.
