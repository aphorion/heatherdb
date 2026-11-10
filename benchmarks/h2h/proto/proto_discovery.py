"""Discovery with the likelihood memory: MDL in real bits (locations with
spread) chooses K; items are assigned to their most probable location."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import task_discovery as t
from likelihood_memory import LikelihoodMemory

Xtr, ytr, Xte, yte, fp = t.load()
rows = []
for k in [5, 10, 15, 20, 30, 40, 60, 80]:
    m = LikelihoodMemory(k, iters=20).fit(Xtr)
    bits = m.description_length_bits()
    s = t.scores(m.posterior(Xtr).argmax(1), ytr, m.posterior(Xte).argmax(1), yte)
    rows.append((bits, k, s))
    print(f"K={k:3d}  DL={bits:,.0f} bits  {s}", flush=True)
bits, k, s = min(rows, key=lambda r: r[0])
print("MDL picks K =", k, s)
