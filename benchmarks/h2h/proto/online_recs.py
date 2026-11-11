"""ONLINE predict-then-store, recommendations. All training interactions are
replayed once in true timestamp order (the real stream, every user
interleaved). Before each item is stored in its user's memory, the user's
current read (time-attention over their history) predicts it; ONE gradient
step on the shared time attention tau; then store. Held-out last items are
never in the stream. tau starts flat (no recency assumed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import data
import task_recs as t
from common import unit

R, ti, eu, hist, fp = t.load()
rng = np.random.default_rng(3)
P = rng.standard_normal((R.shape[0], 512)) / np.sqrt(512)
V = np.asarray(R.T @ P); V -= V.mean(0); V = unit(V)

EDGES = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 10**9])
NB = len(EDGES) - 1

r = data.movielens_1m()
_, u_idx = np.unique(r[:, 0], return_inverse=True)
_, i_idx = np.unique(r[:, 1], return_inverse=True)
# drop each user's held-out last item, then replay by global time
o = np.lexsort((i_idx, r[:, 3], u_idx))
last = np.zeros(len(r), bool)
last[o[np.r_[np.nonzero(np.diff(u_idx[o]))[0], len(o) - 1]]] = True
keep = ~last
stream = np.lexsort((i_idx[keep], u_idx[keep], r[keep, 3]))
su, si = u_idx[keep][stream], i_idx[keep][stream]

# per-user prefix sums of item vectors, so bucket sums are O(NB * d)
hist_items = {}
cums = {}
tau = np.ones(NB)
tau_avg, n_avg = np.zeros(NB), 0   # consolidated (running-average) attention
tau_ema = np.ones(NB)              # consolidated, recent-weighted (EMA, window ~100k writes)
m_, v_ = np.zeros(NB), np.zeros(NB)
lr, b1, b2 = 0.002, 0.9, 0.999
losses, step = [], 0


def buckets(cs, n):
    """bucket sums for a history of length n (lag 1 = newest), via prefix sums."""
    lo = np.minimum(EDGES[:-1], n + 1); hi = np.minimum(EDGES[1:], n + 1)
    # lags [lo, hi) map to positions n-hi+1 .. n-lo   (0-based, inclusive)
    a = n - hi + 1; b = n - lo + 1
    return cs[b] - cs[a]                           # (NB, d)


for u, it in zip(su, si):
    cs = cums.get(u)
    n = 0 if cs is None else len(cs) - 1
    x = V[it]
    if n >= 2:                                     # predict ...
        F = buckets(cs, n)
        fpv = tau @ F
        err = fpv - x
        losses.append(float(err @ err))
        g = 2 * (F @ err)                          # ... learn (one step) ...
        step += 1
        m_ = b1 * m_ + (1 - b1) * g; v_ = b2 * v_ + (1 - b2) * g * g
        tau -= lr * (m_ / (1 - b1 ** step)) / (np.sqrt(v_ / (1 - b2 ** step)) + 1e-8)
        tau = np.maximum(tau, 0)
        n_avg += 1; tau_avg += (tau - tau_avg) / n_avg
        tau_ema += (tau - tau_ema) * 1e-5
        if step % 200000 == 0:
            print(f"  step {step:7d}  running error {np.mean(losses[-200000:]):.4f}  "
                  f"tau/tau0 {np.round(tau / max(tau[0], 1e-12), 2)[:6]}", flush=True)
    new = x if cs is None else cs[-1] + x          # ... then store
    cums[u] = np.vstack([np.zeros((1, V.shape[1])), x[None]]) if cs is None else np.vstack([cs, new[None]])

print("learned time attention (lag buckets, relative):", np.round(tau / tau[0], 3))
for name, tv in [("last iterate", tau), ("consolidated (running average)", tau_avg), ("consolidated (EMA)", tau_ema)]:
    print(f"{name}: attention", np.round(tv / tv[0], 3))
    score = lambda u, tv=tv: V @ unit(tv @ buckets(cums[u], len(cums[u]) - 1))
    print(f"ONLINE error-driven time attention, {name}", t.evaluate(score, R, ti, eu), flush=True)
print("(batch prototype: NDCG@10 0.0626; recency-aware ALS 0.0625)")
