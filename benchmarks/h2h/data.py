"""Public datasets for the head-to-head benchmark.

Every dataset is downloaded from its canonical public source on first use,
cached under ``benchmarks/h2h/data/`` and fingerprinted (SHA-256 of the
arrays actually used), so a rerun on another machine can prove it saw the
same bytes. Nothing needs an account or a licence click-through.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("H2H_DATA", Path(__file__).parent / "data"))
DATA.mkdir(parents=True, exist_ok=True)


def fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Fashion-MNIST (OpenML 40996): 70 000 x 784, 10 classes.
# Used for vector search (an ann-benchmarks dataset) and pattern discovery.
# --------------------------------------------------------------------------
def fashion_mnist():
    cache = DATA / "fashion_mnist.npz"
    if not cache.exists():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(data_id=40996, as_frame=False, return_X_y=True,
                            data_home=str(DATA / "openml"), parser="liac-arff")
        np.savez_compressed(cache, X=X.astype(np.float32), y=y.astype(np.int64))
    z = np.load(cache)
    X, y = z["X"], z["y"]
    # Canonical split: first 60 000 train, last 10 000 test.
    return X[:60000], y[:60000], X[60000:], y[60000:]


# --------------------------------------------------------------------------
# Credit-card fraud (OpenML 1597, ULB / Worldline): 284 807 transactions,
# 492 frauds, 28 PCA features + Time + Amount.
# --------------------------------------------------------------------------
def creditcard():
    cache = DATA / "creditcard.npz"
    if not cache.exists():
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=1597, as_frame=True,
                          data_home=str(DATA / "openml"), parser="auto").frame
        y = df["Class"].astype(str).str.strip("'").astype(int).to_numpy()
        X = df.drop(columns=["Class"]).to_numpy(dtype=np.float64)
        np.savez_compressed(cache, X=X, y=y)
    z = np.load(cache)
    return z["X"], z["y"]


# --------------------------------------------------------------------------
# MovieLens 1M (GroupLens): 1 000 209 ratings, 6 040 users, 3 706 movies.
# --------------------------------------------------------------------------
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def movielens_1m():
    cache = DATA / "ml-1m.zip"
    if not cache.exists():
        with urllib.request.urlopen(ML1M_URL, timeout=120) as r:
            cache.write_bytes(r.read())
    raw = cache.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != ML1M_SHA256:
        raise SystemExit(f"ml-1m.zip sha256 {got} != pinned {ML1M_SHA256}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read("ml-1m/ratings.dat").decode("latin-1")
    rows = np.array([line.split("::") for line in txt.strip().split("\n")],
                    dtype=np.int64)
    return rows  # columns: user, item, rating, timestamp


# --------------------------------------------------------------------------
# Bank marketing (OpenML 1461, UCI): 45 211 customers, 16 attributes,
# target = subscribed to a term deposit (11.7 % positive).
# --------------------------------------------------------------------------
def bank_marketing():
    cache = DATA / "bank_marketing.npz"
    if not cache.exists():
        import pandas as pd
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=1461, as_frame=True,
                          data_home=str(DATA / "openml"), parser="liac-arff").frame
        y = (df["Class"].astype(str) == "2").to_numpy().astype(np.int64)
        X = df.drop(columns=["Class"])
        num = X.select_dtypes("number").to_numpy(np.float64)
        cat = pd.get_dummies(X.select_dtypes(exclude="number").astype(str)).to_numpy(np.float64)
        np.savez_compressed(cache, num=num, cat=cat, y=y)
    z = np.load(cache)
    return z["num"], z["cat"], z["y"]
