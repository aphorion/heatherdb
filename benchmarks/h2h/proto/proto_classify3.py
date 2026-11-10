"""Propensity scoring with selective attention (ALCOVE-style exemplar memory).

Attention per feature is computed from the memory's own stored moments: how
far apart the outcome memories' centres sit on that feature, relative to its
spread (a Fisher ratio). No trained model. Read = exemplar recall with those
attention weights in the distance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import logsumexp

import task_classify as t

Xtr, ytr, Xte, yte, fp = t.load()
m0, m1 = Xtr[ytr == 0], Xtr[ytr == 1]
pooled = (m0.var(0) * len(m0) + m1.var(0) * len(m1)) / len(Xtr) + 1e-9
fisher = (m0.mean(0) - m1.mean(0)) ** 2 / pooled
for name, att in [("uniform attention (engine today)", np.ones_like(fisher)),
                  ("attention = Fisher ratio from stored moments", fisher / fisher.max()),
                  ("attention = sqrt Fisher ratio", np.sqrt(fisher / fisher.max()))]:
    s = np.sqrt(att)
    A, Q = Xtr * s, Xte * s
    for k in [50, 200]:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(A)
        d, i = nn.kneighbors(Q)
        # exemplar recall: softmax over -distance^2 with the neighbourhood's own scale
        sc = d ** 2 / (np.median(d[:, -1:] ** 2) + 1e-12)
        w = np.exp(-sc - logsumexp(-sc, 1, keepdims=True))
        p = (w * ytr[i]).sum(1)
        print(f"{name:46s} k={k:3d}", t.metrics(p, yte), flush=True)
