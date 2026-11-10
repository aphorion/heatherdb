"""Task 2 - Pattern discovery (the job a clustering / segmentation job does).

Dataset   Fashion-MNIST: learn on the 60 000 train images, unlabelled;
          evaluate on the 10 000 test images.
Task      find the groups in the data without labels; assign unseen items.
Metric    NMI and ARI of the test assignment against the hidden labels, and
          purity (each group labelled by its train majority).

HeatherDB  competitive-mode database; the train images are streamed in with
           `write` and the memory grows a codebook. MDL picks how many groups
           the codebook warrants (no K supplied). Unseen items are assigned by
           their most-activated hard location (`batch_analyze`, fast read).
k-means    scikit-learn MiniBatchKMeans, given HeatherDB's K (same budget),
           and k-means given the true number of classes (K=10, an oracle
           choice HeatherDB is not given).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import data
from common import Meter, b64rows, unit


def load():
    Xtr, ytr, Xte, yte = data.fashion_mnist()
    rng = np.random.default_rng(1)
    order = rng.permutation(len(Xtr))  # a stream, not class-sorted
    return unit(Xtr[order]), ytr[order], unit(Xte), yte, data.fingerprint(Xtr, ytr)


def scores(tr_assign, ytr, te_assign, yte):
    # Purity: label each group by its train-set majority, apply to test.
    lab = {}
    for c in np.unique(tr_assign):
        lab[c] = np.bincount(ytr[tr_assign == c]).argmax()
    pred = np.array([lab.get(c, -1) for c in te_assign])
    return {"nmi": float(normalized_mutual_info_score(yte, te_assign)),
            "ari": float(adjusted_rand_score(yte, te_assign)),
            "purity": float(np.mean(pred == yte))}


def heather_assign(h, col, X):
    out = []
    for i in range(0, len(X), 1000):
        r = h.call("POST", col + "/batch_analyze",
                   {"queries": X[i:i + 1000].tolist(), "strategy": "fast"})
        for res in r["results"]:
            acts = res["activated_locations"]
            out.append(max(acts, key=lambda a: a["similarity"])["id"] if acts else -1)
    return np.array(out)


def mdl_bits(X, assign, k_real):
    """Two-part code length in bits: model (k centroids at (d/2)log2 N each),
    index (entropy of the group sizes) and data (Gaussian residual)."""
    N, d = X.shape
    cnt = np.bincount(assign)
    means = np.zeros((len(cnt), d))
    np.add.at(means, assign, X)
    means /= np.maximum(cnt, 1)[:, None]
    var = ((X - means[assign]) ** 2).sum() / (N * d)
    p = cnt[cnt > 0] / N
    return (k_real * (d / 2) * np.log2(N) + N * float(-(p * np.log2(p)).sum())
            + (N * d / 2) * np.log2(2 * np.pi * np.e * var))


def run_heather(h, Xtr, ytr, Xte, yte):
    """The memory learns a codebook from the stream (default competitive
    write); MDL then chooses how many groups the codebook warrants by merging
    its locations (Ward) and keeping the cut with the shortest two-part code.

    The MDL selection runs client-side on the engine's exported codebook,
    because the engine's own `compress` / `mdl_gate` are not yet calibrated
    (engine finding F8). Moving it into `compress` is the fix."""
    from scipy.cluster.hierarchy import fcluster, linkage

    h.create_db("patterns", Xtr.shape[1], precision="f32")
    col = "/db/patterns/collections/images"
    with Meter() as fit:
        for i in range(0, len(Xtr), 1000):
            h.call("POST", col + "/write", {"vectors_b64": b64rows(Xtr[i:i + 1000])})
    locs = h.call("GET", col + "/locations?full=true")["locations"]
    ids = np.array([l["id"] for l in locs])
    A = np.array([l["address"] for l in locs])
    with Meter() as assign:
        te = heather_assign(h, col, Xte)
    tr = heather_assign(h, col, Xtr)
    pos = {d: i for i, d in enumerate(ids)}
    tr_i = np.array([pos[x] for x in tr])
    te_i = np.array([pos[x] for x in te])
    with Meter() as mdl:
        Z = linkage(A, method="ward")
        best = None
        for k in sorted({2, 3, 5, 8, 10, 12, 15, 20, 30, 50, 80, 120, 200, 400, 800}):
            lab = fcluster(Z, k, criterion="maxclust") - 1
            _, ctr = np.unique(lab[tr_i], return_inverse=True)
            bits = mdl_bits(Xtr, ctr, ctr.max() + 1)
            if best is None or bits < best[0]:
                best = (bits, k, ctr.max() + 1)
    _, k, k_real = best
    lab = fcluster(Z, k, criterion="maxclust") - 1
    lab10 = fcluster(Z, 10, criterion="maxclust") - 1
    return {"system": "HeatherDB (codebook + MDL)", "k": int(k_real),
            "k_chosen_by": "MDL", "codebook_locations": len(locs),
            "fit_s": fit.wall + mdl.wall, "assign_s": assign.wall,
            **scores(lab[tr_i], ytr, lab[te_i], yte),
            "at_k10": scores(lab10[tr_i], ytr, lab10[te_i], yte)}


def run_kmeans(Xtr, ytr, Xte, yte, k, label):
    from sklearn.cluster import MiniBatchKMeans

    with Meter() as fit:
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                             batch_size=4096).fit(Xtr)
    with Meter() as assign:
        te = km.predict(Xte)
    return {"system": label, "k": int(k), "k_chosen_by": "operator",
            "fit_s": fit.wall, "assign_s": assign.wall,
            **scores(km.labels_, ytr, te, yte)}
