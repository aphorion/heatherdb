"""Propensity scoring, one collection per outcome: each outcome's customers in
their own memory (K by MDL). Score = which memory is less surprised:
log-odds = surprise_no - surprise_yes + log(prior ratio)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import task_classify as t
from likelihood_memory import mdl_select

Xtr, ytr, Xte, yte, fp = t.load()
mems = {}
for c in (0, 1):
    mems[c], k = mdl_select(Xtr[ytr == c], [1, 2, 5, 10, 20, 50, 100, 200])
    print(f"outcome {c}: {int((ytr == c).sum())} customers, MDL picks K = {k}", flush=True)
logodds = mems[0].surprise(Xte) - mems[1].surprise(Xte) + np.log(ytr.mean() / (1 - ytr.mean()))
print("likelihood read (per-outcome memories)", t.metrics(logodds, yte))
