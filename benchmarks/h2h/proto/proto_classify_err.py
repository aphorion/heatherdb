"""Propensity scoring with ERROR-DRIVEN selective attention (ALCOVE-style).

The memory stays an exemplar store; recall is a softmax over stored customers
with distance  sum_d alpha_d (q_d - e_d)^2.  The attention alpha_d >= 0 is
learned by prediction error: each training customer is recalled from the
OTHER training customers (leave-one-out), and alpha moves to reduce the
log-loss of the recalled outcome. Nothing else is trained. Start = the
moment-based (Fisher) attention."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

import task_classify as t

Xtr, ytr, Xte, yte, fp = t.load()
rng = np.random.default_rng(0)
m0, m1 = Xtr[ytr == 0], Xtr[ytr == 1]
pooled = (m0.var(0) * len(m0) + m1.var(0) * len(m1)) / len(Xtr) + 1e-9
fisher = (m0.mean(0) - m1.mean(0)) ** 2 / pooled
K = 200


def recall(alpha, Q, i_cand, X, y):
    D = ((Q[:, None, :] - X[i_cand]) ** 2 * alpha).sum(-1)
    w = np.exp(-D - logsumexp(-D, 1, keepdims=True))
    return w, (w * y[i_cand]).sum(1), D


def neighbours(alpha, Q, X, k, exclude_self=False):
    s = np.sqrt(alpha)
    nn = NearestNeighbors(n_neighbors=k + exclude_self, n_jobs=-1).fit(X * s)
    _, i = nn.kneighbors(Q * s)
    return i[:, 1:] if exclude_self else i


# Initial attention: Fisher, scaled so a typical neighbour distance is ~1.
alpha = fisher / fisher.max()
idx = rng.choice(len(Xtr), 3000, replace=False)
i0 = neighbours(alpha, Xtr[idx], Xtr, K, exclude_self=True)
d0 = ((Xtr[idx][:, None, :] - Xtr[i0]) ** 2 * alpha).sum(-1)
alpha = alpha / np.median(d0[:, -1])

theta = np.log(alpha + 1e-6)
m_, v_ = np.zeros_like(theta), np.zeros_like(theta)
lr, b1, b2 = 0.05, 0.9, 0.999
for epoch in range(4):                       # re-pick neighbours as attention moves
    qi = rng.choice(len(Xtr), 6000, replace=False)
    Q, yq = Xtr[qi], ytr[qi]
    ic = neighbours(np.exp(theta), Q, Xtr, K, exclude_self=True)
    Dsq = (Q[:, None, :] - Xtr[ic]) ** 2
    yc = ytr[ic]
    for step in range(60):
        a = np.exp(theta)
        D = (Dsq * a).sum(-1)
        w = np.exp(-D - logsumexp(-D, 1, keepdims=True))
        yh = np.clip((w * yc).sum(1), 1e-4, 1 - 1e-4)
        loss = -np.mean(yq * np.log(yh) + (1 - yq) * np.log(1 - yh))
        dL_dyh = (yh - yq) / (yh * (1 - yh)) / len(yq)
        g_s = dL_dyh[:, None] * w * (yc - yh[:, None])          # d loss / d score_j
        g_a = -(g_s[:, :, None] * Dsq).sum((0, 1))              # score_j = -D_j
        g = g_a * a                                              # chain rule through exp(theta)
        it = epoch * 60 + step + 1
        m_ = b1 * m_ + (1 - b1) * g; v_ = b2 * v_ + (1 - b2) * g * g
        theta -= lr * (m_ / (1 - b1 ** it)) / (np.sqrt(v_ / (1 - b2 ** it)) + 1e-8)
    print(f"epoch {epoch}: leave-one-out log-loss {loss:.4f}", flush=True)

alpha = np.exp(theta)
names = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
top = np.argsort(-alpha)[:6]
print("most-attended features (index):", top, "numeric names:", [names[i] for i in top if i < 7])

for label, a in [("Fisher attention (moments)", fisher / fisher.max()), ("error-driven attention", alpha)]:
    ic = neighbours(a, Xte, Xtr, K)
    if label.startswith("Fisher"):
        D = ((Xte[:, None, :] - Xtr[ic]) ** 2 * a).sum(-1)
        D = D / np.median(D[:, -1:])
        w = np.exp(-D - logsumexp(-D, 1, keepdims=True)); p = (w * ytr[ic]).sum(1)
    else:
        _, p, _ = recall(a, Xte, ic, Xtr, ytr)
    print(f"{label:30s}", t.metrics(p, yte), flush=True)
