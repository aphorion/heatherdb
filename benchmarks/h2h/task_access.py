"""Task 7 - Access control (the job a supervised model on ids does).

Dataset   Amazon Employee Access (OpenML 4135): 32,769 access requests,
          9 categorical id columns (resource, manager, role, department,
          title, family, ...) with up to 7,518 distinct values, mostly rare.
          Target: access granted (94 %).
Split     seeded stratified 70/30.
Metric    ROC-AUC (the dataset's standard metric; PR-AUC is uninformative
          at 94 % positives).

HeatherDB  a database declared by columns: the 9 ids as categories and
           `granted` as the number to predict. Requests are written once by
           name in random order; each is predicted from the requests already
           stored, and the miss moves attention and the codes of the values
           involved. Scores come from `predict` with the request's values.
LogReg     scikit-learn LogisticRegression on one-hot ids (the classic strong
           baseline for this dataset), C=1.
GBDT       scikit-learn HistGradientBoostingClassifier on ordinal ids.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import data
from common import Meter

def load():
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=4135, as_frame=True, data_home=str(data.DATA / "openml"),
                      parser="liac-arff").frame
    y = (df["target"].astype(str) == "1").to_numpy().astype(np.int64)
    X = df.drop(columns=["target"]).astype(str)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    return X, y, tr, te, data.fingerprint(X.to_numpy().astype("U"), y)


def metrics(p, y):
    return {"roc_auc": float(roc_auc_score(y, p))}


def run_heather(h, X, y, tr, te):
    fields = [c.lower() for c in X.columns]
    vals = [dict(zip(fields, row)) for row in X.itertuples(index=False)]
    h.create_columns_db("access", {**{f: "category" for f in fields}, "granted": "number"},
                        predict=["granted"])
    col = "/db/access/collections/requests"
    order = tr[np.random.default_rng(0).permutation(len(tr))]
    with Meter() as fit:
        for i in range(0, len(order), 500):
            idx = order[i:i + 500]
            h.call("POST", col + "/rows",
                   {"rows": [{**vals[j], "granted": int(y[j])} for j in idx]})
    p = np.empty(len(te))
    with Meter() as sc:
        for i, j in enumerate(te):
            p[i] = h.call("POST", col + "/predict",
                          {"row": vals[j], "column": "granted", "contributors": 0}
                          )["predicted"]["granted"]
    return {"system": "HeatherDB (learned symbols)", **metrics(p, y[te]),
            "fit_s": fit.wall, "score_us_per_row": sc.wall / len(te) * 1e6}


def run_logreg(X, y, tr, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    with Meter() as fit:
        oh = OneHotEncoder(handle_unknown="ignore").fit(X.iloc[tr])
        m = LogisticRegression(C=1.0, max_iter=5000).fit(oh.transform(X.iloc[tr]), y[tr])
    with Meter() as sc:
        p = m.predict_proba(oh.transform(X.iloc[te]))[:, 1]
    return {"system": "Logistic regression (one-hot)", **metrics(p, y[te]), "fit_s": fit.wall,
            "score_us_per_row": sc.wall / len(te) * 1e6}


def run_gbdt(X, y, tr, te):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    with Meter() as fit:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X.iloc[tr])
        m = HistGradientBoostingClassifier(random_state=0).fit(oe.transform(X.iloc[tr]), y[tr])
    with Meter() as sc:
        p = m.predict_proba(oe.transform(X.iloc[te]))[:, 1]
    return {"system": "Gradient-boosted trees", **metrics(p, y[te]), "fit_s": fit.wall,
            "score_us_per_row": sc.wall / len(te) * 1e6}
