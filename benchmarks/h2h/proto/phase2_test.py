"""Test phase 2 on fraud (surprise, no labels) and propensity (outcome head)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import train_test_split

import data
import task_classify as tc
import task_fraud as tf
from phase2_memory import Phase2Memory

which = sys.argv[1]
spawn = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

if which == "fraud":
    X, y = data.creditcard(); X = X.copy(); X[:, -1] = np.log1p(X[:, -1])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    N = Xtr[ytr == 0]; N = N[np.random.default_rng(4).choice(len(N), tf.N_NORMALS, replace=False)]
    mu, sd = N.mean(0), N.std(0) + 1e-9
    m = Phase2Memory(N.shape[1], spawn=spawn, floor=1e-3)
    for x in (N - mu) / sd:
        m.write(x)
    s = np.concatenate([m.surprise(Q) for Q in np.array_split((Xte - mu) / sd, 50)])
    print(f"fraud spawn={spawn}: K={len(m.n)}", tf.metrics(s, yte), "| LOF 0.485, numpy GMM 0.500, engine 0.199")
else:
    Xtr, ytr, Xte, yte, _ = tc.load()
    d = Xtr.shape[1]
    J = np.hstack([Xtr, np.eye(2)[ytr]])
    m = Phase2Memory(d + 2, heads={"y": [d, d + 1]}, spawn=spawn, floor=0.05)
    for x in J[np.random.default_rng(0).permutation(len(J))]:
        m.write(x)
    Q = np.hstack([Xte, np.zeros((len(Xte), 2))])
    R = np.concatenate([m.recall(q, "y") for q in np.array_split(Q, 30)])[:, d:]
    p = np.clip(R[:, 1], 0, None) / (np.clip(R, 0, None).sum(1) + 1e-12)
    a = np.exp(m.theta["y"][:d])
    print(f"propensity spawn={spawn}: K={len(m.n)}", tc.metrics(p, yte),
          "top features", np.argsort(-a)[:5], "| plain+attention 0.931, GBDT 0.935")
