"""Recency as selective attention over time: the weight of the item k steps
back is how much it predicts the next item beyond chance, measured on
training sequences only (each user's last TRAINING item is the target).
No decay constant. Optionally also attention over the item code's dimensions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import data
import task_recs as t
from common import unit

R, ti, eu, hist, fp = t.load()
rng = np.random.default_rng(3)
P = rng.standard_normal((R.shape[0], 512)) / np.sqrt(512)
V = np.asarray(R.T @ P); V -= V.mean(0); V = unit(V)

# Training sequences for ALL users (held-out last item removed).
r = data.movielens_1m()
_, u_idx = np.unique(r[:, 0], return_inverse=True)
_, i_idx = np.unique(r[:, 1], return_inverse=True)
order = np.lexsort((i_idx, r[:, 3], u_idx))
uo, io = u_idx[order], i_idx[order]
cuts = np.r_[0, np.nonzero(np.diff(uo))[0] + 1, len(uo)]
seqs = [io[a:b - 1] for a, b in zip(cuts[:-1], cuts[1:])]   # drop held-out last

MAXLAG = 300
num, cnt = np.zeros(MAXLAG + 1), np.zeros(MAXLAG + 1)
base = []
for s in seqs:                      # target = last training item, history = before it
    if len(s) < 3:
        continue
    tgt, h = V[s[-1]], s[:-1][::-1][:MAXLAG]   # h[0] = lag 1
    sims = V[h] @ tgt
    num[1:len(h) + 1] += sims; cnt[1:len(h) + 1] += 1
    base.append(V[rng.integers(len(V), size=20)] @ tgt)
chance = float(np.mean(base))
att = np.clip(num[1:] / np.maximum(cnt[1:], 1) - chance, 0, None)   # lag 1..MAXLAG
att[cnt[1:] < 50] = att[cnt[1:] >= 50][-1] if (cnt[1:] >= 50).any() else 0
print("lag attention (lags 1,2,5,10,20,50,100,300):",
      np.round(att[[0, 1, 4, 9, 19, 49, 99, MAXLAG - 1]] / att[0], 3))


def lag_weights(n):
    lags = np.arange(n, 0, -1)                       # oldest .. newest(=1)
    return att[np.minimum(lags, MAXLAG) - 1]


def ev(name, f):
    print(f"{name:52s}", t.evaluate(f, R, ti, eu), flush=True)


actr = lambda n: (np.arange(n, 0, -1).astype(float)) ** -0.5
ev("ACT-R base level (fixed decay 0.5)", lambda u: V @ unit(actr(len(hist[u][0])) @ V[hist[u][0]]))
ev("learned attention over time (lags)", lambda u: V @ unit(lag_weights(len(hist[u][0])) @ V[hist[u][0]]))
