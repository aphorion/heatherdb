"""Fraud with the likelihood read: score = surprise (-log p) under a memory of
normals whose size K is chosen by MDL. No density trick added by hand."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import train_test_split

import data
import task_fraud as t
from likelihood_memory import mdl_select

X, y = data.creditcard()
X = X.copy(); X[:, -1] = np.log1p(X[:, -1])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
normals = Xtr[ytr == 0]
normals = normals[np.random.default_rng(4).choice(len(normals), t.N_NORMALS, replace=False)]
mu, sd = normals.mean(0), normals.std(0) + 1e-9
N, Q = (normals - mu) / sd, (Xte - mu) / sd   # plain standardised features, no cosine tricks

m, k = mdl_select(N, [50, 100, 200, 400, 800], verbose=True)
print("MDL picks K =", k)
print("likelihood read (surprise)", t.metrics(m.surprise(Q), yte))
