"""Which cheap first pass prunes before the int8 scan? GloVe-100 1.18M (and
Fashion-MNIST). For each candidate tier: share of rows whose certified upper
bound can still reach the tier's own k-th best lower bound (= must go on to
the next tier). Bytes/row = what the tier reads. One query at a time.
"""
import sys
import numpy as np

K, NQ = 10, 100
root = "/Users/nguthiru/Work/Aphorion/heather_db/benchmarks/shadow_bench/"
name = sys.argv[1] if len(sys.argv) > 1 else "data_glove"
n, d, nq, mh = map(int, open(root + name + "/shape.txt").read().split())
X = np.fromfile(root + name + "/base.f32", dtype="<f4").reshape(n, d)
Q = np.fromfile(root + name + "/query.f32", dtype="<f4").reshape(nq, d)[:NQ]
V = np.fromfile(root + name + "/axes.f32", dtype="<f4").reshape(mh, d)


def quant(A, bits):
    lim = 2 ** (bits - 1) - 1
    s = np.abs(A).max(1, keepdims=True) / lim
    s[s == 0] = 1
    C = np.clip(np.round(A / s), -lim, lim)
    R = C * s
    return R.astype(np.float32), np.linalg.norm(A - R, axis=1).astype(np.float32), np.linalg.norm(R, axis=1).astype(np.float32)


tiers = {}
for bits in (8, 4, 2):
    R, e, nr = quant(X, bits)
    tiers[f"int{bits} full vector"] = ("full", R, e, nr, d * bits / 8 + 8)
S = X @ V.T                                   # shadow on mh principal axes
res = np.sqrt(np.maximum((X.astype(np.float64) ** 2).sum(1) - (S.astype(np.float64) ** 2).sum(1), 0)).astype(np.float32)
for m in (mh // 2, mh):
    R, e, nr = quant(S[:, :m], 8)
    r_m = np.sqrt(np.maximum((X.astype(np.float64) ** 2).sum(1) - (S[:, :m].astype(np.float64) ** 2).sum(1), 0)).astype(np.float32)
    tiers[f"int8 shadow, {m} axes + leftover"] = ("shadow", R, e, nr, m + 12, m, r_m)

print(f"{name}: n={n:,} d={d}")
for label, t in tiers.items():
    surv = []
    for q in Q:
        qn = float(np.linalg.norm(q))
        if t[0] == "full":
            _, R, e, nr, _ = t
            approx = R @ q
            rad = qn * e + 1e-5 * qn * nr          # x~ quantised, q exact in this test
        else:
            _, R, e, nr, _, m, r_m = t
            qs = V[:m] @ q
            qr = np.sqrt(max(qn * qn - float(qs @ qs), 0))
            approx = R @ qs
            rad = float(np.linalg.norm(qs)) * e + qr * r_m + 1e-5 * qn * (nr + r_m)
        lo, hi = approx - rad, approx + rad
        floor = np.partition(lo, -K)[-K]
        surv.append(np.mean(hi >= floor))
    surv = np.array(surv)
    print(f"  {label:34s} {t[4]:6.1f} B/row   survivors {surv.mean():8.4%}  (p95 {np.percentile(surv, 95):.4%})", flush=True)
