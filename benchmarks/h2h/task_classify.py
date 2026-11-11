"""Task 6 - Tabular classification (the job a supervised ML model does).

Dataset   Bank marketing (UCI, OpenML 1461): 45 211 customers, 16 attributes,
          target = subscribed to a term deposit (11.7 % positive). The
          canonical churn / propensity / lead-scoring shape.
Split     seeded stratified 70/30.
Metric    ROC-AUC and PR-AUC on the test set.

HeatherDB  classification as associative completion, no model: each training
           customer is stored with its outcome attached (address = customer,
           counter = customer + outcome slots). A new customer is read with
           the outcome slots blank; the read fills them in. Score = the
           positive slot's share of the completed outcome.
LogReg     scikit-learn LogisticRegression (the classic baseline).
GBDT       scikit-learn HistGradientBoostingClassifier (the strong default
           for tabular data; the XGBoost/LightGBM family).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import data
from common import Meter, b64rows

BIAS = 30.0
LABEL_W = 1.0


def load():
    num, cat, y = data.bank_marketing()
    num = np.sign(num) * np.log1p(np.abs(num))  # tame heavy tails (balance, duration)
    X = np.hstack([num, cat])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte, data.fingerprint(X, y)


def metrics(p, y):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p))}


def run_heather(h, Xtr, ytr, Xte, yte, k=200, beta=100.0):
    d = Xtr.shape[1]

    def enc(X):
        E = np.hstack([X, np.full((len(X), 1), BIAS)])
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    A = np.hstack([enc(Xtr), np.zeros((len(Xtr), 2))])
    onehot = np.eye(2)[ytr] * LABEL_W
    C = np.hstack([enc(Xtr), onehot])
    h.create_db("propensity", d + 3, competitive=False, k=k, beta=beta)
    col = "/db/propensity/collections/customers"
    with Meter() as fit:
        h.call("POST", col + "/bulk_load",
               {"addresses_b64": b64rows(A), "counters_b64": b64rows(C)})
    Q = np.hstack([enc(Xte), np.zeros((len(Xte), 2))])
    p = np.empty(len(Xte))
    with Meter() as sc:
        for i in range(0, len(Q), 1000):
            r = h.call("POST", col + "/batch_analyze",
                       {"queries": Q[i:i + 1000].tolist(), "strategy": "fast"})
            R = np.array([x["result"] for x in r["results"]])[:, -2:]
            R = np.clip(R, 0, None)
            p[i:i + 1000] = R[:, 1] / (R.sum(1) + 1e-12)
    return {"system": "HeatherDB (associative completion)", **metrics(p, yte),
            "fit_s": fit.wall, "score_us_per_row": sc.wall / len(Q) * 1e6}


def run_logreg(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression

    with Meter() as fit:
        m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    with Meter() as sc:
        p = m.predict_proba(Xte)[:, 1]
    return {"system": "Logistic regression", **metrics(p, yte), "fit_s": fit.wall,
            "score_us_per_row": sc.wall / len(Xte) * 1e6}


def run_gbdt(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import HistGradientBoostingClassifier

    with Meter() as fit:
        m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    with Meter() as sc:
        p = m.predict_proba(Xte)[:, 1]
    return {"system": "Gradient-boosted trees", **metrics(p, yte), "fit_s": fit.wall,
            "score_us_per_row": sc.wall / len(Xte) * 1e6}


def run_heather_sa(h, Xtr, ytr, Xte, yte):
    """HeatherDB with selective attention: a plain database whose `outcome`
    head (two slots) learns feature attention from prediction error while the
    customers are written, once, in random order. Scores come from `recall`."""
    d = Xtr.shape[1]
    h.create_db("propensity_sa", d + 2, competitive=False,
                selective_attention={"heads": {"outcome": [d, d + 1]}, "k": 200,
                                     "warmup": 64})
    col = "/db/propensity_sa/collections/customers"
    rows = np.hstack([Xtr, np.eye(2)[ytr]])
    order = np.random.default_rng(0).permutation(len(rows))
    with Meter() as fit:
        for i in range(0, len(rows), 500):
            h.call("POST", col + "/write", {"vectors_b64": b64rows(rows[order[i:i + 500]])})
    Q = np.hstack([Xte, np.zeros((len(Xte), 2))])
    p = np.empty(len(Q))
    with Meter() as sc:
        for i, q in enumerate(Q):
            r = h.call("POST", col + "/recall",
                       {"query": q.tolist(), "head": "outcome", "contributors": 0})["result"]
            pos, neg = max(r[d + 1], 0.0), max(r[d], 0.0)
            p[i] = pos / (pos + neg + 1e-12)
    return {"system": "HeatherDB (selective attention)", **metrics(p, yte),
            "fit_s": fit.wall, "score_us_per_row": sc.wall / len(Q) * 1e6}


def run_heather_sym(h, Xtr, ytr, Xte, yte):
    """HeatherDB declared by columns: the 9 categorical columns as categories
    (learned codes, no one-hot), the numeric columns as raw numbers (the
    database scales them itself) and `outcome` as the number to predict."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=1461, as_frame=True, data_home=str(data.DATA / "openml"),
                      parser="liac-arff").frame
    y = (df["Class"].astype(str) == "2").to_numpy().astype(np.int64)
    df = df.drop(columns=["Class"])
    num_cols = [c for c in df.columns if df[c].dtype.kind == "f"]
    columns = {c.lower(): ("number" if c in num_cols else "category") for c in df.columns}
    rows = [{c.lower(): (float(v) if c in num_cols else str(v)) for c, v in zip(df.columns, r)}
            for r in df.itertuples(index=False)]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    h.create_columns_db("propensity_sym", {**columns, "outcome": "number"}, predict=["outcome"])
    col = "/db/propensity_sym/collections/customers"
    order = tr[np.random.default_rng(0).permutation(len(tr))]
    with Meter() as fit:
        for i in range(0, len(order), 500):
            idx = order[i:i + 500]
            h.call("POST", col + "/rows", {"rows": [{**rows[j], "outcome": int(y[j])} for j in idx]})
    p = np.empty(len(te))
    with Meter() as sc:
        for i, j in enumerate(te):
            p[i] = h.call("POST", col + "/predict",
                          {"row": rows[j], "column": "outcome", "contributors": 0}
                          )["predicted"]["outcome"]
    return {"system": "HeatherDB (selective attention, learned symbols)", **metrics(p, y[te]),
            "fit_s": fit.wall, "score_us_per_row": sc.wall / len(te) * 1e6}