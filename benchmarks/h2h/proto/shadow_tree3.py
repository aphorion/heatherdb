"""Shadow tree v3: two-level node shadow + warm start (the v2 winners), plus
  ball   per node, a centre and radius in the 24-axis shadow:
         best score anywhere in the ball = q24.c + |q24| r ; bound = min(box, ball)
  leaf   smaller leaves (64 / 32 / 16) so each box and ball is narrower
All bounds certified; every answer checked exact. Memory per query only.
Usage: shadow_tree3.py [dataset]   dataset = fashion (default) | <path to .npy pair prefix>
"""
import heapq
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

K, M1, M2, MH, NQ = 10, 8, 24, 128, 100


def load(name):
    if name == "fashion":
        import task_search as ts
        X, Q, truth, _ = ts.load()
        return X.astype(np.float32), Q[:NQ].astype(np.float32), truth[:NQ]
    X = np.load(name + "_base.npy", mmap_mode="r").astype(np.float32)
    Q = np.load(name + "_query.npy")[:NQ].astype(np.float32)
    truth = np.load(name + "_truth.npy")[:NQ]
    return X, Q, truth


class Node:
    __slots__ = ("lo1", "hi1", "lo2", "hi2", "r2", "bc", "br", "left", "right", "idx", "cen")


class ShadowTree:
    def __init__(self, X, leaf):
        self.X, self.leaf = X, leaf
        n, d = X.shape
        rng = np.random.default_rng(0)
        _, _, Vt = np.linalg.svd(X[rng.choice(n, min(n, 10000), replace=False)].astype(np.float64),
                                 full_matrices=False)
        self.V = Vt[:min(MH, d)].astype(np.float32)
        self.XH = X @ self.V.T                                   # stored shadow (n x 128)
        nrm2 = (X.astype(np.float64) ** 2).sum(1)
        self.XH_r = np.sqrt(np.maximum(nrm2 - (self.XH.astype(np.float64) ** 2).sum(1), 0))
        self.R2 = np.sqrt(np.maximum(nrm2 - (self.XH[:, :M2].astype(np.float64) ** 2).sum(1), 0))
        self.leaves = []
        self.root = self._build(np.arange(n))
        self.LC = np.array([l.cen for l in self.leaves])

    def _build(self, idx):
        nd = Node()
        S1, S2 = self.XH[idx, :M1], self.XH[idx, M1:M2]
        nd.lo1, nd.hi1, nd.lo2, nd.hi2 = S1.min(0), S1.max(0), S2.min(0), S2.max(0)
        nd.r2 = self.R2[idx].max()
        S24 = self.XH[idx, :M2]
        nd.bc = S24.mean(0); nd.br = float(np.linalg.norm(S24 - nd.bc, axis=1).max())
        nd.left = nd.right = nd.idx = nd.cen = None
        if len(idx) <= self.leaf:
            nd.idx = idx
            c = self.XH[idx].mean(0); nd.cen = c / (np.linalg.norm(c) + 1e-12)
            self.leaves.append(nd)
            return nd
        ax = int(np.argmax(nd.hi1 - nd.lo1))
        order = idx[np.argsort(self.XH[idx, ax], kind="stable")]
        mid = len(order) // 2
        nd.left, nd.right = self._build(order[:mid]), self._build(order[mid:])
        return nd

    def search(self, q, ball=True, warm=3):
        X, XH = self.X, self.XH
        qh = self.V @ q
        q1, q2, q24 = qh[:M1], qh[M1:M2], qh[:M2]
        qn2 = float(q @ q)
        qr2 = np.sqrt(max(qn2 - float(q24 @ q24), 0))
        qrh = np.sqrt(max(qn2 - float(qh @ qh), 0))
        n24 = float(np.linalg.norm(q24))
        best = []; seen = set(); cnt = [0, 0, 0, 0]   # touched, 128-tier, exact, nodes
        kth = lambda: best[0] if len(best) == K else -np.inf

        def box(qv, lo, hi):
            return float(np.maximum(qv * lo, qv * hi).sum())

        def ub_node(nd):
            ub = box(q1, nd.lo1, nd.hi1) + box(q2, nd.lo2, nd.hi2)
            if ball:
                ub = min(ub, float(q24 @ nd.bc) + n24 * nd.br)
            return ub + qr2 * nd.r2

        def scan_leaf(nd):
            idx = nd.idx; cnt[0] += len(idx)
            ub1 = XH[idx, :M2] @ q24 + qr2 * self.R2[idx]
            c1 = idx[ub1 > kth()]; cnt[1] += len(c1)
            if len(c1) == 0:
                return
            ub2 = XH[c1] @ qh + qrh * self.XH_r[c1]
            c2 = c1[ub2 > kth()]; cnt[2] += len(c2)
            for s in X[c2] @ q:
                if len(best) < K: heapq.heappush(best, float(s))
                elif s > best[0]: heapq.heapreplace(best, float(s))

        for j in np.argsort(-(self.LC @ qh))[:warm]:
            scan_leaf(self.leaves[j]); seen.add(id(self.leaves[j]))
        heap = [(-ub_node(self.root), 0, self.root)]; c = 1
        while heap:
            nub, _, nd = heapq.heappop(heap)
            if -nub <= kth(): break
            cnt[3] += 1
            if nd.idx is None:
                for ch in (nd.left, nd.right):
                    u = ub_node(ch)
                    if u > kth(): c += 1; heapq.heappush(heap, (-u, c, ch))
            elif id(nd) not in seen:
                scan_leaf(nd)
        return best, cnt


def evaluate(X, Q, truth, leaf, ball):
    n, d = X.shape
    t = ShadowTree(X, leaf)
    tot = np.zeros(4)
    for i in range(len(Q)):
        best, cnt = t.search(Q[i], ball=ball)
        want = np.sort(X[truth[i]].astype(np.float64) @ Q[i].astype(np.float64))[-K:].min()
        assert abs(min(best) - want) < 1e-4, "not exact"
        tot += cnt
    tv, tm, te, nodes = tot / len(Q)
    work = (tv * M2 + tm * MH + te * d + nodes * 4 * M2) / n
    return tv / n, tm / n, te / n, work, d / work


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "fashion"
    X, Q, truth = load(name)
    n, d = X.shape
    print(f"dataset {name}: n={n:,} d={d}   work = multiply-adds per stored vector; flat scan = {d}")
    print(f"{'variant':30s} touched  128-tier  exact    work   vs flat scan")
    for leaf, ball in [(64, False), (64, True), (32, True), (16, True)]:
        tv, tm, te, w, sp = evaluate(X, Q, truth, leaf, ball)
        label = f"leaf {leaf:2d} " + ("box+ball" if ball else "box only")
        print(f"{label:30s} {tv:6.2%}  {tm:7.2%}  {te:6.3%}  {w:6.1f}   {sp:5.1f}x  [exact]", flush=True)
