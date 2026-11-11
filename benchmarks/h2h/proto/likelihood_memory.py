"""Prototype of the proposed engine read: a memory whose locations are
probability components, not bare centres.

Each location l keeps what the engine keeps today, plus one vector:
    mass   pi_l      (write_count; forgetting decays it)
    centre mu_l      (address / normalised counter)
    spread s2_l      (NEW: per-dimension running variance of what it absorbed)

ONE read for every task:
    logit_l(q) = log pi_l - sum_d (q_d - mu_ld)^2 / (2 s2_ld) - 1/2 sum_d log(2 pi s2_ld)
    posterior  = softmax(logit)                     -> what the memory recalls
    surprise   = -logsumexp(logit)  (nats)          -> how unexpected q is
Dimensions the query leaves blank are marginalised out (dropped from the sum),
which is how associative completion reads a missing field.

How many locations: minimum description length in bits, the two-part code
    DL(K) = (#params / 2) log2 N  +  N * surprise / ln 2
(= BIC), so the same likelihood both reads and sizes the memory.

Learning here is batch weighted EM, standing in for the engine's online
competitive writes (online EM is the streaming form of the same update).
Write weights (mass) carry recency or repetition.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

import os

# Recording precision: no location's spread may fall below this fraction of the
# global per-dimension variance. A finite precision is what makes the code
# length a real number of bits (a continuous density on a 0/1 column is
# otherwise unbounded). One constant for every task.
REG = float(os.environ.get("H2H_REG", "1e-3"))


class LikelihoodMemory:
    def __init__(self, k: int, iters: int = 40, seed: int = 0):
        self.k, self.iters, self.seed = k, iters, seed

    # ------------------------------------------------------------ write
    def fit(self, X: np.ndarray, w: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
        w = w / w.sum() * n
        rng = np.random.default_rng(self.seed)
        g = np.average((X - np.average(X, 0, w)) ** 2, 0, w) + 1e-12
        self.floor = REG * g
        # k-means++ seeding on a sample
        p = w / w.sum()
        sample = X[rng.choice(n, 20000, replace=False, p=p)] if n > 20000 else X
        centres = [X[rng.choice(n, p=p)]]
        d2 = ((sample - centres[0]) ** 2).sum(1)
        for _ in range(1, self.k):
            c = sample[rng.choice(len(sample), p=d2 / d2.sum())] if d2.sum() > 0 else sample[rng.integers(len(sample))]
            centres.append(c)
            d2 = np.minimum(d2, ((sample - c) ** 2).sum(1))
        self.mu = np.array(centres, dtype=np.float64)
        self.s2 = np.tile(g, (self.k, 1))
        self.pi = np.full(self.k, 1.0 / self.k)
        for _ in range(self.iters):
            R = self._resp(X)                       # E: posterior per row
            Rw = R * w[:, None]
            nk = Rw.sum(0) + 1e-12                  # M: mass, centre, spread
            self.pi = nk / nk.sum()
            self.mu = (Rw.T @ X) / nk[:, None]
            self.s2 = (Rw.T @ (X * X)) / nk[:, None] - self.mu ** 2
            self.s2 = np.maximum(self.s2, self.floor)
        self.n, self.d = n, d
        self._last_X, self._last_w = X, w
        return self

    # ------------------------------------------------------------- read
    def logits(self, Q: np.ndarray, dims: np.ndarray | None = None) -> np.ndarray:
        mu, s2 = self.mu, self.s2
        if dims is not None:
            Q, mu, s2 = Q[:, dims], mu[:, dims], s2[:, dims]
        inv = 1.0 / s2
        quad = (Q * Q) @ inv.T - 2 * Q @ (mu * inv).T + (mu * mu * inv).sum(1)
        lognorm = -0.5 * np.log(2 * np.pi * s2).sum(1)
        return np.log(self.pi + 1e-300) + lognorm - 0.5 * quad

    def _resp(self, X):
        L = self.logits(X)
        return np.exp(L - logsumexp(L, 1, keepdims=True))

    def posterior(self, Q, dims=None):
        L = self.logits(Q, dims)
        return np.exp(L - logsumexp(L, 1, keepdims=True))

    def surprise(self, Q, dims=None):
        return -logsumexp(self.logits(Q, dims), 1)

    def recall(self, Q, dims=None):
        """Associative completion: posterior-weighted centres (all dims)."""
        return self.posterior(Q, dims) @ self.mu

    # -------------------------------------------------------------- MDL
    def description_length_bits(self) -> float:
        X, w = self._last_X, self._last_w
        nll = float((w * self.surprise(X)).sum())
        params = self.k * (2 * self.d + 1) - 1
        return params / 2 * np.log2(w.sum()) + nll / np.log(2)


def mdl_select(X, ks, w=None, iters=40, seed=0, verbose=False):
    best = None
    for k in ks:
        m = LikelihoodMemory(k, iters, seed).fit(X, w)
        bits = m.description_length_bits()
        if verbose:
            print(f"    K={k:4d}  DL={bits:,.0f} bits", flush=True)
        if best is None or bits < best[0]:
            best = (bits, k, m)
    return best[2], best[1]
