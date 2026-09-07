# Store a procedure

Keep a procedure so a situation recalls it — no name to look up, no key to
remember — and so several instances collapse into one vector that handles a
case none of them covered. The mechanism is in
[Mnemonics](../explanation/mnemonics.md); this page is the recipe.

```python
import numpy as np

D = 1024  # slot budget is ~sqrt(D/32) -- 5 slots here. See step 6.

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

# Circular convolution and correlation: the engine's /vec/bind and
# /vec/unbind in numpy, so this encoding is byte-compatible with the store.
def bind(a, b):
    return normalize(np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D))

def unbind(c, key):
    return np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(key)), n=D)

def sym(name):
    # A deterministic unit vector per symbol. Unrelated names land
    # near-orthogonal, which keeps the sums below separable at read time.
    import hashlib
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return normalize(np.random.default_rng(seed % (2 ** 63)).standard_normal(D))
```

## 1. Encode situation and procedure as role-filler bundles

Both sides must be *structured*: a single opaque embedding of the whole task
has no separable parts, so nothing cancels in step 3 and nothing derives in
step 4. Bind each fact to its role, each command to the step it occupies.

```python
ROLES = ("installer", "tester", "separator", "lang")
ROLE = {k: sym("role:" + k) for k in ROLES}
SLOT = [sym("slot:%d" % i) for i in range(8)]

def situation(facts):
    # The task's discriminative facts, each in its own role. Facts the
    # procedure never uses (lang) are harmless noise; no filtering needed.
    return normalize(sum(bind(ROLE[k], sym(v)) for k, v in facts.items()))

def procedure(steps):
    # The commands, each bound to its step index: order is carried by the slot
    # key, not by the position of anything in memory.
    return normalize(sum(bind(SLOT[i], sym(s)) for i, s in enumerate(steps)))

def eco(inst, test, sep, lang):
    return dict(installer=inst, tester=test, separator=sep, lang=lang)

def steps_of(f):
    # An instance's steps are its own commands -- why they cancel.
    return [f["installer"], f["tester"], f["separator"]]

rust = eco("cargo add", "cargo test", "@", "rust")
s_rust, p_rust = situation(rust), procedure(steps_of(rust))
```

## 2. Store one instance, recalled by situation

Write the relation `unbind(p_rust, s_rust)` into a collection whose address is
the situation, via
[`/db/{db}/collections/procedures/write`](../api/writes.md). Recall then works
by describing the task, never by naming the procedure. One instance already
reproduces its own procedure — and answers only tasks shaped like itself.

## 3. Derive the general form from several instances

Sum the relations. Each instance's own tokens cancel inside its own relation,
so what accumulates is the wiring the instances share.

```python
def relation(facts, steps):
    # Divide the procedure by the situation. A token on both sides (the
    # installer is in the task AND in the command) cancels here, leaving
    # "which role feeds which slot" with no value attached.
    return unbind(procedure(steps), situation(facts))

def mnemonic(instances):
    # The general form: one vector for the whole family.
    return normalize(sum(relation(f, s) for f, s in instances))

FAM = [eco("cargo add", "cargo test", "@", "rust"),
       eco("pip install", "pytest", "==", "python"),
       eco("npm install", "npm test", "@", "node"),
       eco("go get", "go test", "@", "go")]

M = mnemonic([(f, steps_of(f)) for f in FAM])
```

Sixteen instances is where the curve flattens: recovery over a 4-slot schema
runs 49% (2 instances), 74% (4), 92% (8), 97% (16), 98% (64). More never costs
accuracy, so there is no reason to prune a family.

## 4. Apply it in a context it was not built from

Bind the general form onto a new situation and clean up each slot. The new
situation's own tokens flow into the slots.

```python
ruby = eco("bundle add", "rspec", ",", "ruby")
derived = bind(M, situation(ruby))     # a noisy procedure vector

# Cleanup codebook: every command token, including ruby's unseen ones.
labels = sorted({v for f in FAM + [ruby] for v in f.values()})
codebook = np.stack([sym(l) for l in labels])

def read_slot(proc_vec, i, codebook, labels):
    # Unbind the slot key, snap to the nearest known token. This is the
    # cleanup /read performs; done locally it also exposes the score.
    noisy = normalize(unbind(proc_vec, SLOT[i]))
    sims = codebook @ noisy
    j = int(np.argmax(sims))
    return labels[j], float(sims[j])

# Held out of M's construction, every one of ruby's commands is recovered:
# whole-procedure recovery is 100% over 240 held-out members (40 seeds), and
# 0% for the nearest stored instance -- some other ecosystem's commands.
```

Where the codebook is a collection rather than an array, post
`unbind(derived, SLOT[i])` to [`read`](../api/reads.md) with
`"strategy": "fast"` — a single step keeps the distance step 5 needs.

## 5. Gate the result on confidence

A derived procedure is worth as much as the match that produced it. A
situation from outside the family still returns *something*; what says whether
to run it is the first-contact similarity between the slot query and what the
read landed on.

```python
def confident(proc_vec, i, codebook, labels, tau):
    # Below tau the slot is unresolved, and one unresolved slot abstains the
    # whole procedure: a half-derived recipe is worse than none.
    label, score = read_slot(proc_vec, i, codebook, labels)
    return (label, score) if score >= tau else (None, score)
```

Derive `tau` from a held-out sample rather than choosing it — the procedure,
the statistic, and the coverage/error trade are in
[Calibrate a gate](calibrate-a-gate.md). Two of its rules apply directly:
gate on the **first-contact** similarity, not a post-iteration one, and
re-derive the threshold when the codebook grows.

## 6. Budget the slots before the family

The bound is the number of steps in the procedure, not the number of
instances. Recovery over a 16-instance family at D = 1024: 100% at 2–3 slots,
94% at 4, 22% at 6, 0% at 8. Dimension moves the wall — the same 8-slot schema
reads 0% at 1024, 15% at 2048, 96% at 4096, 100% at 8192 — because a schema
fits when `n_slots ≲ √(D/32)`. A longer recipe has two remedies: a larger
[dimension](choose-a-dimension.md), or a chain of shorter mnemonics, which
holds to depth 8 **with a cleanup read between hops** (100% at every depth
measured) against 38% at depth 2 chained naked.

## 7. Add procedures continually

New procedures accumulate; they do not overwrite. Three cases:

**A new family.** The write routes on the situation, so a task unlike anything
stored matches no existing address and starts its own family. Same-family
situations are similar (0.49) where cross-family ones are not (−0.02) — so
the routing needs no labels.

**A new member of an existing family.** Route by situation, add the relation
into the counter at that address. The mnemonic sharpens in place, with no
rebuild and no effect on the other families: independent counters are why a
ten-way stream costs nothing on the classes already learned (91.3% all-seen
after ten rounds, 100% retention of the first).

**A procedure that is now wrong.** Contradiction is the case addition cannot
handle: adding `A → C` on top of `A → B` leaves both, and the read returns
their blend (48% correct inversion). Write the residual instead —
`bind(A, C − read(A))` — which cancels the stale term and installs the new
one (100% inversion, 100% retention of everything not contradicted).

**Two checks.** *Self-regeneration* — ask the mnemonic for the procedures it
was built from; full recovery means the instances are redundant and can be
dropped (100% for families up to 128 members, a 128:1 compression), and a drop
is the signal to split the family or raise the dimension. *The cache baseline*
— score nearest-stored-instance recall on the same held-out cases; a mnemonic
that does not beat it has no cancelling fillers in its encoding.

## Related

- [Mnemonics](../explanation/mnemonics.md) — why cancellation leaves the
  rule · [Calibrate a gate](calibrate-a-gate.md) — the threshold in step 5 ·
  [Build a world model](build-a-world-model.md) — the same write over
  transitions
- [Build an agent memory](build-an-agent-memory.md) ·
  [Choose a dimension](choose-a-dimension.md) ·
  [Compose memories](compose-memories.md) · [bind](../terms/bind.md) ·
  [unbind](../terms/unbind.md) · [online learning](../terms/online-learning.md)
