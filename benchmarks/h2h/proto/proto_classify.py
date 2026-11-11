"""Propensity scoring with the likelihood read: write [customer, outcome]
rows; read a new customer with the outcome dims blank (marginalised); the
recalled outcome dims are the probability. K by MDL. No feature weights by hand."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import task_classify as t
from likelihood_memory import mdl_select

Xtr, ytr, Xte, yte, fp = t.load()
d = Xtr.shape[1]
J = np.hstack([Xtr, np.eye(2)[ytr]])          # outcome stored as two slots
m, k = mdl_select(J, [2, 5, 10, 20, 50, 100, 200, 400], verbose=True)
print("MDL picks K =", k)
R = m.recall(Xte, dims=np.arange(d))[:, d:]
p = np.clip(R[:, 1], 0, None) / (np.clip(R, 0, None).sum(1) + 1e-12)
print("likelihood read (completion)", t.metrics(p, yte))
