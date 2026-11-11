"""Task 5 - Fraud / anomaly detection (the job an outlier-detection model does).

Dataset   Credit-card fraud (ULB/Worldline, OpenML 1597): 284 807 card
          transactions, 492 frauds (0.17 %), 28 PCA features + Amount.
Split     seeded stratified 50/50. Detectors see NO fraud labels: they learn
          "normal" from a 50 000-transaction sample of training normals.
Metric    ROC-AUC and PR-AUC (average precision) on the full test half;
          recall at a 1 % false-positive budget (the usual review-queue cut).

HeatherDB  `bulk_load` the normals (each its own attractor, k=1 read);
           score = 1 - cosine(transaction, fast-read reconstruction). The
           encoding is standardised features plus a constant bias coordinate
           c=100, unit-normalised: for c >> |x|, 1 - cos(a, b) ~ |a-b|^2/2c^2,
           so the cosine read measures Euclidean distance to normal.
IForest    scikit-learn IsolationForest (300 trees), same normals.
LOF        scikit-learn LocalOutlierFactor (novelty, 20 neighbours), same normals.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import data
from common import Meter, b64rows

N_NORMALS = 50_000


def load():
    """Standardised features (by the training normals' mean and spread),
    unchanged otherwise: every detector sees the same numbers."""
    X, y = data.creditcard()
    X = X.copy()
    X[:, -1] = np.log1p(X[:, -1])  # Amount is heavy-tailed
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y,
                                          random_state=0)
    normals = Xtr[ytr == 0]
    rng = np.random.default_rng(4)
    normals = normals[rng.choice(len(normals), N_NORMALS, replace=False)]
    mu, sd = normals.mean(0), normals.std(0) + 1e-9
    return (normals - mu) / sd, (Xte - mu) / sd, yte, data.fingerprint(X, y)


def unit_with_bias(A):
    """The nearest-distance encoding: standardised features plus a bias
    coordinate c=100, unit-normalised, so 1 - cosine ~ |a-b|^2 / 2c^2."""
    E = np.hstack([A, np.full((len(A), 1), 100.0)])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def metrics(score, y):
    # Recall when the alert budget is 1 % of genuine traffic.
    thr = np.quantile(score[y == 0], 0.99)
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "recall@1%fpr": float(np.mean(score[y == 1] > thr))}


BEFORE_LABEL = "HeatherDB (nearest distance)"


def run_heather(h, normals, Xte, yte):
    """Before: raw distance to the nearest stored normal (a k=1 read)."""
    normals, Xte = unit_with_bias(normals), unit_with_bias(Xte)
    h.create_db("fraud", normals.shape[1], competitive=False, k=1)
    col = "/db/fraud/collections/normal_card_activity"
    with Meter() as fit:
        h.call("POST", col + "/bulk_load",
               {"addresses_b64": b64rows(normals), "counters_b64": b64rows(normals)})
    Qn = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    score = np.empty(len(Xte))
    with Meter() as sc:
        for i in range(0, len(Xte), 1000):
            r = h.call("POST", col + "/batch_analyze",
                       {"queries": Xte[i:i + 1000].tolist(), "strategy": "fast"})
            R = np.array([x["result"] for x in r["results"]])
            R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            score[i:i + 1000] = 1.0 - np.sum(R * Qn[i:i + 1000], axis=1)
    return {"system": "HeatherDB", **metrics(score, yte), "fit_s": fit.wall,
            "score_us_per_txn": sc.wall / len(Xte) * 1e6}


def run_heather_v2(h, normals, Xte, yte, k=50):
    """Novelty relative to the neighbourhood: the stored normals are the
    memory; each transaction is scored by `novelty` (its distance to its k
    nearest normals relative to how spread out those normals usually are)."""
    h.create_db("fraud_novelty", normals.shape[1], competitive=False)
    col = "/db/fraud_novelty/collections/normal_card_activity"
    with Meter() as fit:
        h.call("POST", col + "/bulk_load",
               {"addresses_b64": b64rows(normals), "counters_b64": b64rows(normals)})
    score = np.empty(len(Xte))
    with Meter() as sc:
        for i in range(0, len(Xte), 1000):
            r = h.call("POST", col + "/novelty",
                       {"queries": Xte[i:i + 1000].tolist(), "k": k})
            score[i:i + 1000] = [x["novelty"] for x in r["results"]]
    return {"system": "HeatherDB (novelty)", **metrics(score, yte), "fit_s": fit.wall,
            "score_us_per_txn": sc.wall / len(Xte) * 1e6}


def run_iforest(normals, Xte, yte):
    from sklearn.ensemble import IsolationForest

    with Meter() as fit:
        m = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(normals)
    with Meter() as sc:
        s = -m.score_samples(Xte)
    return {"system": "Isolation Forest", **metrics(s, yte), "fit_s": fit.wall,
            "score_us_per_txn": sc.wall / len(Xte) * 1e6}


def run_lof50(normals, Xte, yte):
    """LOF at the same neighbour count HeatherDB's novelty uses."""
    return run_lof(normals, Xte, yte, k=50)


def run_lof(normals, Xte, yte, k=20):
    from sklearn.neighbors import LocalOutlierFactor

    with Meter() as fit:
        m = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(normals)
    with Meter() as sc:
        s = -m.score_samples(Xte)
    return {"system": "LOF" if k == 20 else f"LOF (k={k})", **metrics(s, yte), "fit_s": fit.wall,
            "score_us_per_txn": sc.wall / len(Xte) * 1e6}
