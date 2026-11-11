"""ONLINE predict-then-store, propensity. Customers arrive one at a time, in
random order, once. For each: read it with the outcome blank, compare the
recall to the true outcome, take ONE gradient step on the attention, then
store it. No batch passes, no leave-one-out sweeps, no Fisher start:
attention starts uniform."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp

import task_classify as t

Xtr, ytr, Xte, yte, fp = t.load()
rng = np.random.default_rng(0)
order = rng.permutation(len(Xtr))
K, WARM = 200, 500
d = Xtr.shape[1]

# uniform attention, scaled so a typical neighbour distance is ~1 (from the warm-up rows)
W = Xtr[order[:WARM]]
dd = ((W[:, None, :] - W[None, :, :]) ** 2).sum(-1)
theta = np.full(d, np.log(1.0 / np.quantile(dd[dd > 0], 0.01)))
m_, v_ = np.zeros(d), np.zeros(d)
lr, b1, b2 = 0.01, 0.9, 0.999

mem_X = np.empty_like(Xtr); mem_y = np.empty_like(ytr); n = 0
for j in order[:WARM]:
    mem_X[n], mem_y[n] = Xtr[j], ytr[j]; n += 1

losses = []
for step, j in enumerate(order[WARM:], 1):
    q, yq = Xtr[j], ytr[j]
    a = np.exp(theta)
    Dsq = (mem_X[:n] - q) ** 2                    # predict ...
    D = Dsq @ a
    ic = np.argpartition(D, K)[:K]
    s = -D[ic]
    w = np.exp(s - logsumexp(s))
    yh = w @ mem_y[ic]
    losses.append((yh - yq) ** 2)
    g_s = 2 * (yh - yq) * w * (mem_y[ic] - yh)                 # squared error (engine form)
    g = -(g_s @ Dsq[ic]) * a
    m_ = b1 * m_ + (1 - b1) * g; v_ = b2 * v_ + (1 - b2) * g * g
    theta -= lr * (m_ / (1 - b1 ** step)) / (np.sqrt(v_ / (1 - b2 ** step)) + 1e-8)
    mem_X[n], mem_y[n] = q, yq; n += 1           # ... then store
    if step % 5000 == 0:
        print(f"  {n:6d} stored, running sq-error {np.mean(losses[-5000:]):.4f}", flush=True)

a = np.exp(theta)
P = np.empty(len(Xte))
for i in range(0, len(Xte), 500):
    Qb = Xte[i:i + 500]
    D = ((Qb * Qb) @ a)[:, None] - 2 * (Qb * a) @ Xtr.T + ((Xtr * Xtr) @ a)[None]   # same distance, fast form
    ic = np.argpartition(D, K, axis=1)[:, :K]
    s = -np.take_along_axis(D, ic, 1)
    w = np.exp(s - logsumexp(s, 1, keepdims=True))
    P[i:i + 500] = (w * ytr[ic]).sum(1)
print("most-attended features:", np.argsort(-a)[:6])
print("ONLINE error-driven attention, squared-error loss", t.metrics(P, yte))
print("(batch prototype: 0.931 / 0.605; GBDT 0.935 / 0.629)")
