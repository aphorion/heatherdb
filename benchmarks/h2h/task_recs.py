"""Task 3 - Recommendations, and Task 4 - Freshness (new data -> served).

Dataset   MovieLens 1M (GroupLens). Implicit feedback: a rating is a view.
Split     leave-last-out by timestamp: each user's most recent rating is held
          out; everything else is training data. Evaluated on a seeded
          sample of users (same users for every system).
Task      rank the whole catalogue (seen items excluded) for each user.
Metric    HR@10 and NDCG@10 of the held-out item.

HeatherDB  storage-native, no training step: an item's vector is a fixed
           random projection of the users who watched it (d=128, computed
           from training data); each user's history is written into their
           own collection; the user's `fingerprint` ranks the catalogue.
Popularity most-watched items.
ItemKNN    cosine item-item neighbourhood over the interaction matrix.
ALS        `implicit` AlternatingLeastSquares, 64 factors, 15 iterations
           (the standard matrix-factorisation recommender).
Raw sum    the same item vectors HeatherDB gets, summed with no memory: shows
           what the associative memory adds beyond the encoding.

Freshness  the cost to make a brand-new user's history servable:
           HeatherDB = write that user's collection (one request);
           ALS = refit the model (the usual nightly retrain), timed.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

import data
from common import Meter, b64rows, unit

DIM = 128
N_EVAL = 2000
K = 10
HIST_TS: dict = {}   # user -> training timestamps, aligned with hist[u]


def load():
    r = data.movielens_1m()
    users, u_idx = np.unique(r[:, 0], return_inverse=True)
    items, i_idx = np.unique(r[:, 1], return_inverse=True)
    ts, rating = r[:, 3], r[:, 2]
    # Last interaction per user (ties broken by item id for determinism).
    order = np.lexsort((i_idx, ts, u_idx))
    last = np.zeros(len(r), dtype=bool)
    last[order[np.r_[np.nonzero(np.diff(u_idx[order]))[0], len(order) - 1]]] = True
    tr = ~last
    nu, ni = len(users), len(items)
    R = sp.csr_matrix((np.ones(tr.sum()), (u_idx[tr], i_idx[tr])), shape=(nu, ni))
    test_item = np.full(nu, -1)
    test_item[u_idx[last]] = i_idx[last]
    rng = np.random.default_rng(2)
    eval_users = np.sort(rng.choice(nu, N_EVAL, replace=False))
    # Per-user ordered history with ratings, for the write stream.
    hist = {}
    for u in eval_users:
        m = (u_idx == u) & tr
        o = np.argsort(ts[m], kind="stable")
        hist[u] = (i_idx[m][o], rating[m][o])
        HIST_TS[u] = ts[m][o]
    return R, test_item, eval_users, hist, data.fingerprint(r)


def evaluate(score_fn, R, test_item, eval_users):
    hits, ndcg = [], []
    for u in eval_users:
        s = np.asarray(score_fn(u), dtype=np.float64).ravel().copy()
        s[R[u].indices] = -np.inf
        target = s[test_item[u]]
        rank = int(np.sum(s > target))
        hits.append(rank < K)
        ndcg.append(1.0 / np.log2(rank + 2) if rank < K else 0.0)
    return {"hr@10": float(np.mean(hits)), "ndcg@10": float(np.mean(ndcg))}


def item_vectors(R):
    """Storage-native item code: a fixed random projection of each item's
    audience. No fitting, no gradient: one sparse matmul."""
    rng = np.random.default_rng(3)
    P = rng.standard_normal((R.shape[0], DIM)) / np.sqrt(DIM)
    V = (R.T @ P)
    V = V - V.mean(axis=0, keepdims=True)  # centre (see docs: center-your-vectors)
    return unit(V)


def reps(rating):
    return 3 if rating >= 5 else (2 if rating >= 4 else 1)


# ---------------------------------------------------------------- HeatherDB
def run_heather(h, R, test_item, eval_users, hist):
    h.create_db("recs", DIM, precision="f32")
    with Meter() as enc:
        V = item_vectors(R)
    write_lat = []
    with Meter() as load_t:
        for u in eval_users:
            items, ratings = hist[u]
            rows = np.repeat(items, [reps(x) for x in ratings])
            t0 = time.perf_counter()
            for i in range(0, len(rows), 4000):
                h.call("POST", f"/db/recs/collections/user_{u}/write",
                       {"vectors_b64": b64rows(V[rows[i:i + 4000]])})
            write_lat.append(time.perf_counter() - t0)

    def score(u):
        fp = h.call("GET", f"/db/recs/collections/user_{u}/fingerprint")["fingerprint"]
        return V @ unit(np.array(fp))

    with Meter() as serve:
        res = evaluate(score, R, test_item, eval_users)
    raw = evaluate(lambda u: V @ unit(V[hist[u][0]].sum(0)), R, test_item, eval_users)
    return ({"system": "HeatherDB", **res, "fit_s": enc.wall, "load_s": load_t.wall,
             "serve_ms_per_user": serve.wall / len(eval_users) * 1000},
            {"system": "Raw sum (same vectors, no memory)", **raw},
            {"new_user_ready_ms_p50": float(np.median(write_lat) * 1000),
             "new_user_ready_ms_p95": float(np.percentile(write_lat, 95) * 1000)})


# ---------------------------------------------------------------- baselines
def run_popularity(R, test_item, eval_users):
    pop = np.asarray(R.sum(0)).ravel()
    return {"system": "Popularity", **evaluate(lambda u: pop, R, test_item, eval_users)}


def run_itemknn(R, test_item, eval_users):
    with Meter() as fit:
        n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
        Rn = R @ sp.diags(1 / n)
        S = (Rn.T @ Rn).toarray()
        np.fill_diagonal(S, 0)
    res = evaluate(lambda u: R[u] @ S, R, test_item, eval_users)
    return {"system": "ItemKNN (cosine)", **res, "fit_s": fit.wall}


def fit_als(R):
    from implicit.als import AlternatingLeastSquares

    m = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.05,
                                random_state=0)
    m.fit(R.tocsr() * 20.0, show_progress=False)  # alpha=20 confidence
    return m


def run_als(R, test_item, eval_users):
    with Meter() as fit:
        m = fit_als(R)
    uf, itf = np.asarray(m.user_factors), np.asarray(m.item_factors)
    res = evaluate(lambda u: itf @ uf[u], R, test_item, eval_users)
    return {"system": "ALS (implicit, 64 factors)", **res, "fit_s": fit.wall,
            "retrain_to_add_new_user_s": fit.wall}


# ------------------------------------------------- selective attention (engine)
SA_DIM = 512


def item_vectors_sa(R):
    rng = np.random.default_rng(3)
    P = rng.standard_normal((R.shape[0], SA_DIM)) / np.sqrt(SA_DIM)
    V = np.asarray(R.T @ P)
    V = V - V.mean(axis=0, keepdims=True)
    return unit(V)


def run_heather_sa(h, R, test_item, eval_users, hist):
    """HeatherDB with selective attention over time: one plain collection per
    user; every training interaction is written in global timestamp order (each
    run of one user's consecutive interactions is one request). The database
    learns its recency curve from prediction error while it is written; the
    `next` read ranks the catalogue. No decay constant is set."""
    h.create_db("recs_sa", SA_DIM, competitive=False, precision="f32",
                selective_attention={"time": True})
    V = item_vectors_sa(R)
    # Global stream of the evaluated users' training interactions, by time.
    events = []
    for u in eval_users:
        items, _ = hist[u]
        ts = HIST_TS[u]
        events += [(t_, u, it) for t_, it in zip(ts, items)]
    events.sort(key=lambda e: (e[0], e[1]))
    with Meter() as load_t:
        i = 0
        while i < len(events):
            j = i
            while j < len(events) and events[j][1] == events[i][1]:
                j += 1
            u = events[i][1]
            rows = V[[e[2] for e in events[i:j]]]
            h.call("POST", f"/db/recs_sa/collections/user_{u}/write",
                   {"vectors_b64": b64rows(rows)})
            i = j

    def score(u):
        nxt = h.call("GET", f"/db/recs_sa/collections/user_{u}/next")["result"]
        return V @ np.array(nxt)

    with Meter() as serve:
        res = evaluate(score, R, test_item, eval_users)
    learned = h.call("GET", "/db/recs_sa/selective-attention")["time"]
    return {"system": "HeatherDB (selective attention)", **res, "load_s": load_t.wall,
            "serve_ms_per_user": serve.wall / len(eval_users) * 1000,
            "learned_time_weights": [round(w / learned["weights"][0], 3) for w in learned["weights"]],
            "time_updates": learned["updates"]}


# ---------------------------------------- recency-aware incumbents (fairness)
ACTR_DECAY = 0.5


def recency_matrix(R):
    """Training interactions weighted by age^-0.5 (ACT-R), age 1 = newest."""
    r = data.movielens_1m()
    _, u_idx = np.unique(r[:, 0], return_inverse=True)
    _, i_idx = np.unique(r[:, 1], return_inverse=True)
    order = np.lexsort((i_idx, r[:, 3], u_idx))
    uo, io = u_idx[order], i_idx[order]
    starts = np.r_[0, np.nonzero(np.diff(uo))[0] + 1]
    lens = np.diff(np.r_[starts, len(uo)])
    pos = np.arange(len(uo)) - np.repeat(starts, lens)
    n_u = np.repeat(lens, lens)
    keep = pos < n_u - 1
    age = np.maximum((n_u - 1) - pos, 1)
    w = age.astype(float) ** -ACTR_DECAY
    return sp.csr_matrix((w[keep], (uo[keep], io[keep])), shape=R.shape)


def run_itemknn_recency(R, test_item, eval_users):
    Rw = recency_matrix(R)
    with Meter() as fit:
        n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
        Rn = R @ sp.diags(1 / n)
        S = (Rn.T @ Rn).toarray()
        np.fill_diagonal(S, 0)
    res = evaluate(lambda u: Rw[u] @ S, R, test_item, eval_users)
    return {"system": "ItemKNN + recency", **res, "fit_s": fit.wall}


def run_als_recency(R, test_item, eval_users):
    from implicit.als import AlternatingLeastSquares

    Rw = recency_matrix(R)
    with Meter() as fit:
        m = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.05,
                                    random_state=0)
        m.fit(Rw.tocsr() * 20.0, show_progress=False)
    uf, itf = np.asarray(m.user_factors), np.asarray(m.item_factors)
    res = evaluate(lambda u: itf @ uf[u], R, test_item, eval_users)
    return {"system": "ALS + recency (time-decayed confidence)", **res, "fit_s": fit.wall,
            "retrain_to_add_new_user_s": fit.wall}
