"""Certified shadow filters for exact top-k (cosine), Fashion-MNIST 60k x 784.

For each filter: the fraction of stored vectors that must still be scored
exactly after the filter, with the answer guaranteed identical to brute force.
  global   f32 shadow, one worst-case bound (today's write scan)
  int8 / 1-bit shadows with PER-VECTOR residual bounds  |q.x - q.x~| <= |q| |x - x~|
  balls    k-means cells: skip a cell if  q.c + r  <  current k-th best
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import task_search as ts

K = 10
X, Q, truth, _ = ts.load()
Q = Q[:100]; truth = truth[:100]
n, d = X.shape


def certify(approx, radius, name):
    """approx: (q, n) shadow scores; radius: (n,) per-vector bound |x - x~| (q is unit)."""
    kept = []
    for i in range(len(Q)):
        a, r = approx[i], radius
        lower = a - r                                  # true score >= lower
        t = np.partition(lower, -K)[-K]                # k-th best guaranteed score
        cand = np.nonzero(a + r >= t)[0]               # anything that could still beat it
        exact = X[cand] @ Q[i]
        top = cand[np.argsort(-exact)[:K]]
        assert set(top) == set(truth[i]), name
        kept.append(len(cand) / n)
    kept = np.array(kept)
    print(f"{name:42s} exact work {kept.mean():7.3%} of vectors (p95 {np.percentile(kept,95):.3%})  [answer exact]")


# 1. today's f32 shadow, global worst-case bound
X32 = X.astype(np.float32)
eps = float(np.abs(X - X32).max()) * np.sqrt(d)      # crude global bound, as a single number
certify((Q.astype(np.float32) @ X32.T).astype(float), np.full(n, eps), "f32 shadow, global bound (today)")

# 2. int8 shadow, per-vector residual bound
s = np.abs(X).max(1, keepdims=True) / 127
X8 = np.round(X / s).astype(np.int8)
Xq = X8 * s
certify(Q @ Xq.T, np.linalg.norm(X - Xq, axis=1), "int8 shadow, per-vector bound (4x smaller)")

# 3. 1-bit sign shadow, scaled, per-vector residual bound
B = np.sign(X - X.mean(0)) ; B[B == 0] = 1
mu = X.mean(0)
scale = np.einsum("ij,ij->i", X - mu, B) / d
Xb = mu + scale[:, None] * B
certify(Q @ Xb.T, np.linalg.norm(X - Xb, axis=1), "1-bit shadow + mean, per-vector bound (32x)")

# 4. balls: k-means cells as the index; prune whole cells by centre score + radius
for C in (64, 256, 1024):
    km = MiniBatchKMeans(C, random_state=0, n_init=1, batch_size=8192).fit(X)
    lab = km.labels_; cent = km.cluster_centers_
    rad = np.zeros(C)
    np.maximum.at(rad, lab, np.linalg.norm(X - cent[lab], axis=1))
    members = [np.nonzero(lab == c)[0] for c in range(C)]
    kept = []
    for i in range(len(Q)):
        cs = cent @ Q[i]
        order = np.argsort(-cs)
        best = np.full(K, -np.inf); scanned = 0
        for c in order:
            if cs[c] + rad[c] < best.min():          # nothing in this ball can enter the top k
                continue
            sc = X[members[c]] @ Q[i]; scanned += len(sc)
            best = np.sort(np.concatenate([best, sc]))[-K:]
        kept.append(scanned / n)
        assert np.isclose(best.min(), np.sort(X[truth[i]] @ Q[i]).min()), "balls"
    print(f"{'balls, k-means C=' + str(C):42s} exact work {np.mean(kept):7.3%} of vectors (p95 {np.percentile(kept,95):.3%})  [answer exact]")
