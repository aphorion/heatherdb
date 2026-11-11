"""Phase 2 prototype: a COMPETITIVE memory shaped by selective attention.

Location l:  mass by age bucket m_l[b], centre mu_l, per-feature spread s2_l.
Score (read AND routing):
    logit_l(q) = log(sum_b tau_b m_l[b])
                 - 1/2 sum_{d not masked} alpha_d (q_d - mu_ld)^2 / s2_ld
                 - 1/2 sum_{d not masked} log s2_ld
Write x (predict -> learn -> route):
    1. heads: recall x with each head's fields blank; one Adam step on log alpha
       toward predicting them (gradient through the softmax over locations).
    2. route with the SAME attended score: if the best location explains x
       (attended per-dimension distance < SPAWN), join it (online EM update
       of mass, centre, spread); otherwise spawn a new location at x.
Attention therefore shapes which rows share a location: what the memory learns.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

EDGES = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 10**9])
NB = len(EDGES) - 1


class Adam:
    def __init__(self, n, lr):
        self.m, self.v, self.t, self.lr = np.zeros(n), np.zeros(n), 0, lr

    def step(self, p, g):
        self.t += 1
        self.m = .9 * self.m + .1 * g
        self.v = .999 * self.v + .001 * g * g
        return p - self.lr * (self.m / (1 - .9 ** self.t)) / (np.sqrt(self.v / (1 - .999 ** self.t)) + 1e-8)


class Phase2Memory:
    def __init__(self, d, heads=None, k=50, spawn=4.0, floor=0.05, lr=0.01, warmup=64,
                 time=False, cap=None):
        self.d, self.k, self.spawn, self.floor, self.warmup = d, k, spawn, floor, warmup
        self.heads = {h: np.isin(np.arange(d), dims) for h, dims in (heads or {}).items()}
        self.theta = {h: np.zeros(d) for h in self.heads}
        self.opt = {h: Adam(d, lr) for h in self.heads}
        self.time = time
        self.tau = np.ones(NB)
        self.cap = cap
        self.mu = np.zeros((0, d)); self.sq = np.zeros((0, d)); self.n = np.zeros(0)
        self.times = []                     # write times per location (for age buckets)
        self.t = 0
        self.gvar = None; self._buf = []

    # ---- helpers -------------------------------------------------------
    def s2(self):
        v = self.sq / self.n[:, None] - self.mu ** 2
        return np.maximum(v, self.floor * self.gvar)

    def logmass(self):
        if not self.time:
            return np.log(self.n)
        out = np.empty(len(self.n))
        for i, ts in enumerate(self.times):
            lag = self.t - np.asarray(ts)
            b = np.searchsorted(EDGES, np.maximum(lag, 1), side="right") - 1
            out[i] = np.log(max((self.tau[b]).sum(), 1e-12))
        return out

    def alpha(self, head):
        if head is None:
            return np.ones(self.d), np.zeros(self.d, bool)
        return np.exp(self.theta[head]), self.heads[head]

    def logits(self, Q, head=None):
        a, mask = self.alpha(head)
        keep = ~mask
        s2 = self.s2()[:, keep]; mu = self.mu[:, keep]; Qk = np.atleast_2d(Q)[:, keep]; ak = a[keep]
        inv = ak / s2
        quad = (Qk * Qk) @ inv.T - 2 * Qk @ (mu * inv).T + (mu * mu * inv).sum(1)
        return self.logmass()[None] - 0.5 * quad - 0.5 * np.log(s2).sum(1)[None], quad / keep.sum()

    # ---- read ----------------------------------------------------------
    def recall(self, Q, head=None):
        L, _ = self.logits(Q, head)
        w = np.exp(L - logsumexp(L, 1, keepdims=True))
        return w @ self.mu

    def surprise(self, Q):
        L, _ = self.logits(Q, None)
        return -logsumexp(L, 1)

    # ---- write ---------------------------------------------------------
    def _learn_head(self, h, x):
        a, mask = self.alpha(h)
        keep = ~mask
        L, _ = self.logits(x, h)
        L = L[0]
        idx = np.argsort(-L)[: self.k]
        w = np.exp(L[idx] - logsumexp(L[idx]))
        pred = w @ self.mu[idx][:, mask]
        err = pred - x[mask]
        g_s = 2 * w * ((self.mu[idx][:, mask] - pred) @ err)        # dLoss/dlogit
        s2 = self.s2()[idx]
        dq = (x[None] - self.mu[idx]) ** 2 / s2                       # d logit/d alpha = -1/2 dq
        g = np.zeros(self.d)
        g[keep] = (g_s @ (-0.5 * dq[:, keep])) * a[keep]
        self.theta[h] = np.clip(self.opt[h].step(self.theta[h], g), -30, 30)
        self.theta[h][mask] = 0

    def write(self, x):
        x = np.asarray(x, float)
        self.t += 1
        if self.gvar is None:                         # warm-up: collect rows for global variance
            self._buf.append(x)
            if len(self._buf) < self.warmup:
                return
            B = np.array(self._buf); self.gvar = B.var(0) + 1e-9
            for r in B:
                self._spawn(r)
            return
        for h in self.heads:
            self._learn_head(h, x)
        head = next(iter(self.heads), None)           # routing uses the (first) head's attention
        L, dist = self.logits(x, head)
        j = int(np.argmax(L[0]))
        if dist[0, j] > self.spawn and (self.cap is None or len(self.n) < self.cap):
            self._spawn(x)
        else:
            self.n[j] += 1; self.sq[j] += x * x
            self.mu[j] += (x - self.mu[j]) / self.n[j]
            self.times[j].append(self.t)

    def _spawn(self, x):
        self.mu = np.vstack([self.mu, x]); self.n = np.append(self.n, 1.0)
        # a new location starts with the spread of a quarter of the global variance
        self.sq = np.vstack([self.sq, x * x + 0.25 * self.gvar]); self.times.append([self.t])
