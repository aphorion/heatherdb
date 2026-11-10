"""Task 1 - Vector search (the job a vector database does).

Dataset   Fashion-MNIST, 60 000 x 784 indexed, 1 000 test-set queries.
Task      top-10 nearest neighbours by cosine similarity.
Truth     exact brute-force top-10 (numpy, float64).
Metric    recall@10; single-query latency p50/p95 over HTTP; index build time.

HeatherDB  plain-mode database, every vector a document; `documents/query`.
Qdrant     the real server (Docker, pinned image), cosine, default HNSW
           (m=16, ef_construct=100), REST API, default search params.
"""

from __future__ import annotations

import subprocess
import time

import numpy as np
import requests

import data
from common import Meter, b64rows, free_port, pct, unit

QDRANT_IMAGE = "qdrant/qdrant:v1.12.4"
N_QUERIES = 1000
K = 10


def load(n_index: int | None = None):
    Xtr, _, Xte, _ = data.fashion_mnist()
    if n_index:
        Xtr = Xtr[:n_index]
    rng = np.random.default_rng(0)
    Q = Xte[rng.choice(len(Xte), N_QUERIES, replace=False)]
    Xu, Qu = unit(Xtr), unit(Q)
    truth = np.argsort(-(Qu @ Xu.T), axis=1)[:, :K]
    return Xu, Qu, truth, data.fingerprint(Xtr, Q)


def recall(found: list[list[int]], truth: np.ndarray) -> float:
    return float(np.mean([len(set(f[:K]) & set(t)) / K for f, t in zip(found, truth)]))


# ---------------------------------------------------------------- HeatherDB
def run_heather(h, Xu, Qu, truth):
    h.create_db("search", Xu.shape[1], competitive=False, precision="f32")
    col = "/db/search/collections/items"
    ids = np.empty(len(Xu), dtype=np.int64)
    with Meter() as build:
        for i in range(0, len(Xu), 2000):
            r = h.call("POST", col + "/write",
                       {"vectors_b64": b64rows(Xu[i:i + 2000]),
                        "metadata": [{"i": int(j)} for j in range(i, min(i + 2000, len(Xu)))]})
            ids[i:i + len(r["ids"])] = r["ids"]
    id2row = {int(d): j for j, d in enumerate(ids)}
    found, lat = [], []
    for q in Qu:
        t0 = time.perf_counter()
        r = h.call("POST", col + "/documents/query", {"query": q.tolist(), "n": K})
        lat.append(time.perf_counter() - t0)
        found.append([id2row[x["id"]] for x in r["results"]])
    return {"system": "HeatherDB", "recall@10": recall(found, truth),
            "build_s": build.wall, "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "qps_single_client": len(lat) / sum(lat)}


# ------------------------------------------------------------------- Qdrant
def docker_mem_bytes(name: str) -> int:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format",
                          "{{.MemUsage}}", name], capture_output=True, text=True).stdout
    used = out.split("/")[0].strip()
    num = float("".join(c for c in used if c.isdigit() or c == "."))
    unit_ = used.lstrip("0123456789.").strip().upper()
    mult = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3,
            "KB": 1e3, "MB": 1e6, "GB": 1e9}[unit_]
    return int(num * mult)


def run_qdrant(Xu, Qu, truth):
    port, name = free_port(), f"h2h-qdrant-{int(time.time())}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name,
                    "-p", f"127.0.0.1:{port}:6333", QDRANT_IMAGE],
                   check=True, capture_output=True)
    base = f"http://127.0.0.1:{port}"
    s = requests.Session()
    try:
        for _ in range(300):
            try:
                if s.get(base + "/readyz", timeout=1).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        idle = docker_mem_bytes(name)
        s.put(base + "/collections/items", json={
            "vectors": {"size": Xu.shape[1], "distance": "Cosine"}}).raise_for_status()
        with Meter() as build:
            for i in range(0, len(Xu), 1000):
                pts = [{"id": int(j), "vector": Xu[j].astype(np.float32).tolist()}
                       for j in range(i, min(i + 1000, len(Xu)))]
                s.put(base + "/collections/items/points?wait=true",
                      json={"points": pts}).raise_for_status()
            # Wait for HNSW indexing to finish, as a production load would.
            while s.get(base + "/collections/items").json()["result"]["status"] != "green":
                time.sleep(0.2)
        peak = docker_mem_bytes(name)
        found, lat = [], []
        for q in Qu:
            t0 = time.perf_counter()
            r = s.post(base + "/collections/items/points/search",
                       json={"vector": q.astype(np.float32).tolist(), "limit": K}).json()
            lat.append(time.perf_counter() - t0)
            found.append([p["id"] for p in r["result"]])
        peak = max(peak, docker_mem_bytes(name))
        disk = subprocess.run(["docker", "exec", name, "du", "-sb", "/qdrant/storage"],
                              capture_output=True, text=True).stdout.split()[0]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return {"system": f"Qdrant ({QDRANT_IMAGE.split(':')[1]})",
            "recall@10": recall(found, truth), "build_s": build.wall,
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "qps_single_client": len(lat) / sum(lat),
            "mem_bytes": peak, "mem_idle_bytes": idle, "disk_bytes": int(disk)}
