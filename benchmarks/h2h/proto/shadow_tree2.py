"""Shadow tree, ablation of three tightenings (all certified exact):
  1 two-level node shadow: box on axes [0,8) (tree splits) AND box on [8,24),
    residual only beyond axis 24:  ub = box1 + box2 + |q_r24| * max|x_r24|
  2 warm start: before traversal, exactly score the leaves whose centroid has
    the best 128-axis shadow score, so the k-th best is high from the start
  3 int8 128-axis tier with per-vector quantisation error in the bound
Cost model (multiply-adds per stored vector, averaged): tier dims x fraction reaching it.
"""
import heapq
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import task_search as ts

K, LEAF, M1, M2, MH = 10, 64, 8, 24, 128
X, Q, truth, _ = ts.load()
Q, truth = Q[:100], truth[:100]
n, d = X.shape
rng = np.random.default_rng(0)
_, _, Vt = np.linalg.svd(X[rng.choice(n, 10000, replace=False)], full_matrices=False)
XH = X @ Vt[:MH].T
XH_r = np.sqrt(np.maximum(1 - (XH ** 2).sum(1), 0))
S1, S2 = XH[:, :M1], XH[:, M1:M2]
R1 = np.sqrt(np.maximum(1 - (S1 ** 2).sum(1), 0))
R2 = np.sqrt(np.maximum(1 - (XH[:, :M2] ** 2).sum(1), 0))
# int8 128-axis shadow with per-vector error radius
sc = np.abs(XH).max(1, keepdims=True) / 127
XH8 = np.round(XH / sc).astype(np.int8)
XHq = XH8 * sc
E8 = np.linalg.norm(XH - XHq, axis=1)


class Node:
    __slots__ = ("lo1", "hi1", "lo2", "hi2", "r1", "r2", "left", "right", "idx", "cen")


def build(idx):
    nd = Node()
    nd.lo1, nd.hi1 = S1[idx].min(0), S1[idx].max(0)
    nd.lo2, nd.hi2 = S2[idx].min(0), S2[idx].max(0)
    nd.r1, nd.r2 = R1[idx].max(), R2[idx].max()
    nd.left = nd.right = nd.idx = None
    nd.cen = None
    if len(idx) <= LEAF:
        nd.idx = idx
        c = XH[idx].mean(0); nd.cen = c / (np.linalg.norm(c) + 1e-12)
        return nd
    ax = int(np.argmax(nd.hi1 - nd.lo1))
    order = idx[np.argsort(S1[idx, ax])]
    mid = len(order) // 2
    nd.left, nd.right = build(order[:mid]), build(order[mid:])
    return nd


root = build(np.arange(n))
leaves = []
def collect(nd):
    if nd.idx is not None: leaves.append(nd)
    else: collect(nd.left); collect(nd.right)
collect(root)
LC = np.array([l.cen for l in leaves])


def box(q, lo, hi):
    return np.maximum(q * lo, q * hi).sum()


def run(two_level, warm, int8):
    tv = tm = te = nodes = 0.0
    for i in range(len(Q)):
        q = Q[i]; qh = Vt[:MH] @ q
        q1, q2 = qh[:M1], qh[M1:M2]
        qr1 = np.sqrt(max(1 - (q1 ** 2).sum(), 0))
        qr2 = np.sqrt(max(1 - (qh[:M2] ** 2).sum(), 0))
        qrh = np.sqrt(max(1 - (qh ** 2).sum(), 0))
        best = []; seen_leaf = set()
        kth = lambda: best[0] if len(best) == K else -np.inf

        def ub_node(nd):
            if two_level:
                return box(q1, nd.lo1, nd.hi1) + box(q2, nd.lo2, nd.hi2) + qr2 * nd.r2
            return box(q1, nd.lo1, nd.hi1) + qr1 * nd.r1

        def scan_leaf(nd):
            nonlocal tv, tm, te
            idx = nd.idx; tv += len(idx)
            if two_level:
                ub1 = S1[idx] @ q1 + S2[idx] @ q2 + qr2 * R2[idx]
            else:
                ub1 = S1[idx] @ q1 + qr1 * R1[idx]
            c1 = idx[ub1 > kth()]; tm += len(c1)
            if len(c1) == 0:
                return
            if int8:
                ub2 = XHq[c1] @ qh + np.linalg.norm(qh) * E8[c1] + qrh * XH_r[c1]
            else:
                ub2 = XH[c1] @ qh + qrh * XH_r[c1]
            c2 = c1[ub2 > kth()]; te += len(c2)
            for s in X[c2] @ q:
                if len(best) < K: heapq.heappush(best, s)
                elif s > best[0]: heapq.heapreplace(best, s)

        if warm:                                   # exactly score the 3 most promising leaves first
            for j in np.argsort(-(LC @ qh))[:3]:
                scan_leaf(leaves[j]); seen_leaf.add(id(leaves[j]))
        heap = [(-ub_node(root), 0, root)]; cnt = 1
        while heap:
            nub, _, nd = heapq.heappop(heap)
            if -nub <= kth(): break
            nodes += 1
            if nd.idx is None:
                for ch in (nd.left, nd.right):
                    u = ub_node(ch)
                    if u > kth(): cnt += 1; heapq.heappush(heap, (-u, cnt, ch))
            elif id(nd) not in seen_leaf:
                scan_leaf(nd)
        assert np.isclose(min(best), np.sort(X[truth[i]] @ q).min()), "not exact"
    nq = len(Q)
    tv, tm, te, nodes = tv / nq / n, tm / nq / n, te / nq / n, nodes / nq
    low = M2 if two_level else M1
    mid_cost = MH / 4 if int8 else MH                 # int8 multiply-adds count as a quarter
    flops = tv * low + tm * mid_cost + te * d + nodes * 2 * low / n
    return tv, tm, te, flops


print(f"{'variant':34s} touched  128-tier  exact    work/vector  speedup vs scan(784)")
for name, a, b, c in [("baseline tree (m=8)", False, False, False),
                      ("+1 two-level node shadow", True, False, False),
                      ("+1+2 warm start", True, True, False),
                      ("+1+2+3 int8 128-axis tier", True, True, True)]:
    tv, tm, te, fl = run(a, b, c)
    print(f"{name:34s} {tv:6.2%}  {tm:7.2%}  {te:6.3%}  {fl:9.1f}     {d / fl:5.1f}x   [exact]", flush=True)
