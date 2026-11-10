"""Shadow triangulation for exact top-k cosine (unit vectors), Fashion-MNIST 60k x 784.

Exact work = share of vectors whose bound cannot rule them out (then scored in f64).
Every answer is checked identical to brute force.
  pivots   store d(x, p) for m pivots;  d(q,x) >= max_p |d(q,p) - d(x,p)|
  subspace store x's shadow on m PCA axes + |residual|;
           q.x in [q_s.x_s - |q_r||x_r|, q_s.x_s + |q_r||x_r|]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import task_search as ts

K = 10
X, Q, truth, _ = ts.load()
Q, truth = Q[:100], truth[:100]
n, d = X.shape


def check(lower_score, upper_score, name, per_vec_bytes):
    """Bounds on the true dot score per (query, vector)."""
    kept = []
    for i in range(len(Q)):
        t = np.partition(lower_score[i], -K)[-K]          # k-th best guaranteed score
        cand = np.nonzero(upper_score[i] >= t)[0]
        top = cand[np.argsort(-(X[cand] @ Q[i]))[:K]]
        assert set(top) == set(truth[i]), name
        kept.append(len(cand) / n)
    kept = np.array(kept)
    print(f"{name:34s} {per_vec_bytes:5d} B/vector  exact work {kept.mean():7.3%}  (p95 {np.percentile(kept, 95):6.3%})", flush=True)


# --- pivots: on unit vectors, distance^2 = 2 - 2 score ---------------------------
rng = np.random.default_rng(0)
for m in (8, 32, 128):
    P = X[rng.choice(n, m, replace=False)]
    dx = np.sqrt(np.maximum(2 - 2 * X @ P.T, 0))           # (n, m) stored
    dq = np.sqrt(np.maximum(2 - 2 * Q @ P.T, 0))           # (q, m)
    # one query at a time: (n, m) at most, never (q, n, m)
    lo_d = np.empty((len(Q), n)); hi_d = np.empty((len(Q), n))
    for i in range(len(Q)):
        lo_d[i] = np.abs(dx - dq[i]).max(1)                 # lower bound on distance
        hi_d[i] = (dx + dq[i]).min(1)
    check(1 - hi_d ** 2 / 2, 1 - lo_d ** 2 / 2, f"pivots m={m}", 4 * m)
    del lo_d, hi_d, dx

# --- subspace shadow + residual -------------------------------------------------
mu = 0  # cosine on unit vectors: project without centring so the bound is exact
_, _, Vt = np.linalg.svd(X[rng.choice(n, 10000, replace=False)], full_matrices=False)
for m in (8, 32, 64, 128):
    B = Vt[:m]                                              # m orthonormal axes
    xs, qs = X @ B.T, Q @ B.T
    xr = np.sqrt(np.maximum(1 - (xs ** 2).sum(1), 0))        # |x residual| (unit vectors)
    qr = np.sqrt(np.maximum(1 - (qs ** 2).sum(1), 0))
    s = qs @ xs.T
    slack = qr[:, None] * xr[None]
    check(s - slack, s + slack, f"subspace shadow m={m}", 4 * (m + 1))
