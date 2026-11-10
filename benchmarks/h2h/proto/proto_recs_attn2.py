"""(1) Error-driven attention over time: lag-bucket weights fitted to PREDICT
the next item (non-negative least squares on training sequences), so
redundant recent items are not counted twice. (2) Representation ceiling:
the same read over ALS's learned item vectors instead of the random code."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import nnls

import data
import task_recs as t
from common import unit

R, ti, eu, hist, fp = t.load()
rng = np.random.default_rng(3)
P = rng.standard_normal((R.shape[0], 512)) / np.sqrt(512)
V = np.asarray(R.T @ P); V -= V.mean(0); V = unit(V)

r = data.movielens_1m()
_, u_idx = np.unique(r[:, 0], return_inverse=True)
_, i_idx = np.unique(r[:, 1], return_inverse=True)
order = np.lexsort((i_idx, r[:, 3], u_idx))
uo, io = u_idx[order], i_idx[order]
cuts = np.r_[0, np.nonzero(np.diff(uo))[0] + 1, len(uo)]
seqs = [io[a:b - 1] for a, b in zip(cuts[:-1], cuts[1:])]

EDGES = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 10**9])   # log-spaced lag buckets


def bucket_feats(Vm, h_newest_first):
    lags = np.arange(1, len(h_newest_first) + 1)
    b = np.searchsorted(EDGES, lags, side="right") - 1
    F = np.zeros((len(EDGES) - 1, Vm.shape[1]))
    np.add.at(F, b, Vm[h_newest_first])
    return F


def fit_attention(Vm):
    rows, ys = [], []
    for s in seqs:
        if len(s) < 3:
            continue
        F = bucket_feats(Vm, s[:-1][::-1])
        rows.append(F.T); ys.append(Vm[s[-1]])
    A = np.vstack(rows); y = np.concatenate(ys)
    w, _ = nnls(A, y)
    return w


def score_with(Vm, w):
    return lambda u: Vm @ unit(w @ bucket_feats(Vm, hist[u][0][::-1]))


def ev(name, f):
    print(f"{name:58s}", t.evaluate(f, R, ti, eu), flush=True)


w = fit_attention(V)
print("error-driven bucket weights (lag 1,2,3-4,5-7,...):", np.round(w / w.max(), 3))
ev("random code + error-driven time attention", score_with(V, w))

from implicit.als import AlternatingLeastSquares
m = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.05, random_state=0)
m.fit(R.tocsr() * 20.0, show_progress=False)
Vals = unit(np.asarray(m.item_factors))
actr = lambda n: (np.arange(n, 0, -1).astype(float)) ** -0.5
ev("ALS item vectors + ACT-R read (representation ceiling)", lambda u: Vals @ unit(actr(len(hist[u][0])) @ Vals[hist[u][0]]))
w2 = fit_attention(Vals)
ev("ALS item vectors + error-driven time attention", score_with(Vals, w2))
