"""Recommendations with a brain-style base level (ACT-R): each past item's
weight is age^-0.5, age = interactions since it (0.5 = ACT-R's standard decay,
taken from the human-memory literature, not tuned here). The same weights are
given to the incumbents, so recency is not HeatherDB's private advantage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp

import data
import task_recs as t
from common import unit

DECAY = 0.5
R, ti, eu, hist, fp = t.load()

# Recency-weighted interaction matrix for ALL users (training interactions only).
r = data.movielens_1m()
_, u_idx = np.unique(r[:, 0], return_inverse=True)
_, i_idx = np.unique(r[:, 1], return_inverse=True)
order = np.lexsort((i_idx, r[:, 3], u_idx))
uo, io = u_idx[order], i_idx[order]
starts = np.r_[0, np.nonzero(np.diff(uo))[0] + 1]
lens = np.diff(np.r_[starts, len(uo)])
pos = np.arange(len(uo)) - np.repeat(starts, lens)
n_u = np.repeat(lens, lens)
keep = pos < n_u - 1                              # drop each user's held-out last item
age = (n_u - 1) - pos                             # 1 = most recent training item
w_all = age.astype(float) ** -DECAY
Rw = sp.csr_matrix((w_all[keep], (uo[keep], io[keep])), shape=R.shape)


def base_level(u):
    it = hist[u][0]
    return (len(it) - np.arange(len(it))).astype(float) ** -DECAY


def ev(name, f):
    print(f"{name:52s}", t.evaluate(f, R, ti, eu), flush=True)


# HeatherDB-style: storage-native item code (d=512), base-level-weighted recall
rng = np.random.default_rng(3)
P = rng.standard_normal((R.shape[0], 512)) / np.sqrt(512)
V = np.asarray(R.T @ P); V -= V.mean(0); V = unit(V)
ev("memory, no recency (centroid, d=512)", lambda u: V @ unit(V[hist[u][0]].sum(0)))
ev("memory + base level (ACT-R d=0.5)", lambda u: V @ unit(base_level(u) @ V[hist[u][0]]))

# Incumbents, without and with the same recency weights
n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
Rn = R @ sp.diags(1 / n); S = (Rn.T @ Rn).toarray(); np.fill_diagonal(S, 0)
ev("ItemKNN", lambda u: R[u] @ S)
ev("ItemKNN + base level", lambda u: Rw[u] @ S)

from implicit.als import AlternatingLeastSquares
for name, M in [("ALS", R), ("ALS + base level (time-decayed confidence)", Rw)]:
    best = None
    for f in (64, 128):
        m = AlternatingLeastSquares(factors=f, iterations=15, regularization=0.05, random_state=0)
        m.fit(M.tocsr() * 20.0, show_progress=False)
        uf, itf = np.asarray(m.user_factors), np.asarray(m.item_factors)
        res = t.evaluate(lambda u: itf @ uf[u], R, ti, eu)
        print(f"{name + f' f={f}':52s}", res, flush=True)
