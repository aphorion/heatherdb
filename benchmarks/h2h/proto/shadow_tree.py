"""Shadow + index: a k-d tree over each vector's low-dimensional shadow (its
coordinates on the first m PCA axes), searched best-first with a CERTIFIED
upper bound per node:
    ub(node) = max over the node's box of q_s . x_s  +  |q_r| * max |x_r| in node
Search stops when no unvisited node can beat the current k-th best exact score.
In reached leaves: per-vector low-shadow bound -> 128-axis shadow bound -> exact.
Reports how much of the collection a query touches; answers checked exact.
Memory: O(n * 128) floats; no (queries x n x m) arrays.
"""
import heapq
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import task_search as ts

K, LEAF, M_HI = 10, 64, 128
X, Q, truth, _ = ts.load()
Q, truth = Q[:100], truth[:100]
n, d = X.shape
rng = np.random.default_rng(0)
_, _, Vt = np.linalg.svd(X[rng.choice(n, 10000, replace=False)], full_matrices=False)
XH = X @ Vt[:M_HI].T                                        # 128-axis shadow (stored)
XH_r = np.sqrt(np.maximum(1 - (XH ** 2).sum(1), 0))


class Node:
    __slots__ = ("lo", "hi", "rmax", "left", "right", "idx")


def build(idx, S, R):
    node = Node()
    node.lo, node.hi = S[idx].min(0), S[idx].max(0)
    node.rmax = R[idx].max()
    node.left = node.right = node.idx = None
    if len(idx) <= LEAF:
        node.idx = idx
        return node
    ax = int(np.argmax(node.hi - node.lo))
    order = idx[np.argsort(S[idx, ax])]
    mid = len(order) // 2
    node.left, node.right = build(order[:mid], S, R), build(order[mid:], S, R)
    return node


def box_ub(qs, node):
    return np.maximum(qs * node.lo, qs * node.hi).sum()


for m in (8, 16, 32):
    S = XH[:, :m]
    R = np.sqrt(np.maximum(1 - (S ** 2).sum(1), 0))         # residual beyond the m-axis shadow
    t0 = time.time(); root = build(np.arange(n), S, R); tb = time.time() - t0
    touched, mid, exact, nodes = [], [], [], []
    for i in range(len(Q)):
        q = Q[i]; qh = Vt[:M_HI] @ q
        qs, qr = qh[:m], np.sqrt(max(1 - (qh[:m] ** 2).sum(), 0))
        qhr = np.sqrt(max(1 - (qh ** 2).sum(), 0))
        best = []                                            # min-heap of top-K exact scores
        kth = lambda: best[0] if len(best) == K else -np.inf
        heap = [(-(box_ub(qs, root) + qr * root.rmax), 0, root)]
        cnt = 1; t_v = t_m = t_e = n_nodes = 0
        while heap:
            neg_ub, _, node = heapq.heappop(heap)
            if -neg_ub <= kth():
                break
            n_nodes += 1
            if node.idx is None:
                for ch in (node.left, node.right):
                    ub = box_ub(qs, ch) + qr * ch.rmax
                    if ub > kth():
                        cnt += 1; heapq.heappush(heap, (-ub, cnt, ch))
                continue
            idx = node.idx; t_v += len(idx)
            ub1 = S[idx] @ qs + qr * R[idx]                    # low-shadow bound per vector
            c1 = idx[ub1 > kth()]; t_m += len(c1)
            if len(c1) == 0:
                continue
            sh = XH[c1] @ qh
            ub2 = sh + qhr * XH_r[c1]                          # 128-axis shadow bound
            c2 = c1[ub2 > kth()]; t_e += len(c2)
            for s in X[c2] @ q:
                if len(best) < K:
                    heapq.heappush(best, s)
                elif s > best[0]:
                    heapq.heapreplace(best, s)
        assert np.isclose(min(best), np.sort(X[truth[i]] @ q).min()), "not exact"
        touched.append(t_v / n); mid.append(t_m / n); exact.append(t_e / n); nodes.append(n_nodes)
    print(f"tree on m={m:2d} shadow (build {tb:4.1f}s): touched {np.mean(touched):6.2%} of vectors "
          f"(p95 {np.percentile(touched, 95):6.2%}), 128-axis bound on {np.mean(mid):6.2%}, "
          f"exact on {np.mean(exact):6.3%}, nodes visited {np.mean(nodes):.0f}  [exact]", flush=True)
print("(flat subspace shadow m=128 for comparison: touches 100%, exact on 2.6%)")
