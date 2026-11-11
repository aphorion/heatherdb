#!/usr/bin/env python3
"""HeatherDB vs Qdrant as a vector database: exact top-10 cosine search.

    HEATHER_BIN=.../target/release/heather .venv/bin/python vectordb_bench.py [fashion|glove|all]
    -> results/vectordb.json

Datasets
  fashion  Fashion-MNIST (OpenML 40996): 60,000 x 784 indexed, 1,000 test
           queries; truth = exact top-10 by cosine (numpy, f64).
  glove    GloVe-100-angular (ann-benchmarks): 1,183,514 x 100 indexed,
           first 1,000 test queries; truth = the published neighbours.
           Vectors unit-normalised (angular = cosine).

Systems (each in a fresh process or container, measured alone)
  HeatherDB memory   plain database, f32 storage, vectors resident in RAM.
  HeatherDB shadow   the same with residency "shadow": an int8 shadow in
                     RAM, exact vectors read from storage when needed.
  Qdrant default     qdrant/qdrant:v1.12.4, cosine, default HNSW
                     (m=16, ef_construct=100), default search parameters.
  Qdrant ef=256      the same collection searched with hnsw_ef=256
                     (higher recall, slower).

Measures (all over HTTP, JSON bodies on both sides)
  recall@10, single-client latency p50/p95 and queries/s, throughput with
  8 concurrent clients (each pass repeated 3 times: median and range), load time (writes + index build, until the first
  query can run at full speed), peak memory (sampled every 0.25 s from
  start to end, so load-time peaks count), steady memory after queries,
  and disk use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

QDRANT_IMAGE = "qdrant/qdrant:v1.12.4"
K = 10
NQ = 1000
CLIENTS = 8
SHADOW_DATA = HERE.parent / "shadow_bench"


# ------------------------------------------------------------------ data
def load_dataset(name: str):
    if name == "fashion":
        import task_search as ts

        X, Q, truth, _ = ts.load()
        return X.astype(np.float32), Q[:NQ].astype(np.float32), truth[:NQ]
    d = SHADOW_DATA / "data_glove"
    n, dim, nq, _ = map(int, (d / "shape.txt").read_text().split())
    X = np.fromfile(d / "base.f32", dtype="<f4").reshape(n, dim)
    Q = np.fromfile(d / "query.f32", dtype="<f4").reshape(nq, dim)[:NQ]
    truth = np.fromfile(d / "truth.u32", dtype="<u4").reshape(nq, K)[:NQ]
    return X, Q, truth


def recall(found, truth):
    return float(np.mean([len(set(f[:K]) & set(int(t) for t in tr)) / K for f, tr in zip(found, truth)]))


def pct(xs, q):
    return float(np.percentile(np.asarray(xs) * 1000.0, q))


def qps(single, conc):
    """Median and range of the repeated throughput passes."""
    return {"qps_1_client": float(np.median(single)),
            "qps_1_client_range": [float(min(single)), float(max(single))],
            f"qps_{CLIENTS}_clients": float(np.median(conc)),
            f"qps_{CLIENTS}_clients_range": [float(min(conc)), float(max(conc))]}


class Sampler:
    """Samples a memory reading every 0.25 s on a background thread."""

    def __init__(self, read):
        self.read, self.peak, self._stop = read, 0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.read())
            except Exception:
                pass
            time.sleep(0.25)

    def stop(self):
        self._stop.set()
        self.t.join()
        return self.peak


REPEATS = 3


def run_queries(one, Q):
    """Single-client latency (every query, REPEATS passes), then CLIENTS
    concurrent clients' throughput, REPEATS times. A laptop's background load
    swings one run by tens of percent, so throughput is reported as the
    median with its range, and latency percentiles pool every pass."""
    one(Q[0])  # warm-up
    lat, found, single = [], None, []
    for _ in range(REPEATS):
        res, t_pass = [], time.perf_counter()
        for q in Q:
            t = time.perf_counter()
            res.append(one(q))
            lat.append(time.perf_counter() - t)
        single.append(len(Q) / (time.perf_counter() - t_pass))
        found = found or res
    conc = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(CLIENTS) as ex:
            list(ex.map(one, Q))
        conc.append(len(Q) / (time.perf_counter() - t0))
    return found, lat, single, conc


# -------------------------------------------------------------- HeatherDB
def bench_heather(X, Q, truth, residency):
    h = common.Heather()
    sampler = Sampler(lambda: h.ps.memory_info().rss)
    try:
        h.create_db("vec", X.shape[1], competitive=False, precision="f32", residency=residency)
        col = "/db/vec/collections/items"
        ids = np.empty(len(X), dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(0, len(X), 2000):
            r = h.call("POST", col + "/write", {
                "vectors_b64": common.b64rows(X[i:i + 2000].astype(np.float64)),
                "metadata": [{"i": int(j)} for j in range(i, min(i + 2000, len(X)))]})
            ids[i:i + len(r["ids"])] = r["ids"]
        load_s = time.perf_counter() - t0
        id2row = {int(d): j for j, d in enumerate(ids)}
        url = h.base + col + "/documents/query"
        local = threading.local()

        def one(q):
            s = getattr(local, "s", None)
            if s is None:
                s = local.s = requests.Session()
                s.headers.update(h.s.headers)
            r = s.post(url, json={"query": q.astype(np.float64).tolist(), "n": K}, timeout=600)
            r.raise_for_status()
            return [id2row[x["id"]] for x in r.json()["results"]]

        found, lat, single, conc = run_queries(one, Q)
        steady = h.ps.memory_info().rss
        disk = h.disk_bytes()
    finally:
        peak = sampler.stop()
        h.close()
    return {"system": f"HeatherDB ({residency} residency)", "recall@10": recall(found, truth),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), **qps(single, conc), "load_s": load_s,
            "peak_mem_mib": peak / 2**20, "steady_mem_mib": steady / 2**20, "disk_mib": disk / 2**20}


# ----------------------------------------------------------------- Qdrant
def docker_mem(name):
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         capture_output=True, text=True).stdout
    used = out.split("/")[0].strip()
    num = float("".join(c for c in used if c.isdigit() or c == "."))
    unit = used.lstrip("0123456789.").strip().upper()
    return int(num * {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3,
                      "KB": 1e3, "MB": 1e6, "GB": 1e9}[unit])


def bench_qdrant(X, Q, truth):
    port, name = common.free_port(), f"vdb-qdrant-{int(time.time())}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:6333",
                    QDRANT_IMAGE], check=True, capture_output=True)
    base = f"http://127.0.0.1:{port}"
    s = requests.Session()
    out = []
    try:
        for _ in range(300):
            try:
                if s.get(base + "/readyz", timeout=1).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.1)
        sampler = Sampler(lambda: docker_mem(name))
        s.put(base + "/collections/items",
              json={"vectors": {"size": X.shape[1], "distance": "Cosine"}}).raise_for_status()
        t0 = time.perf_counter()
        for i in range(0, len(X), 1000):
            pts = [{"id": int(j), "vector": X[j].tolist()} for j in range(i, min(i + 1000, len(X)))]
            s.put(base + "/collections/items/points?wait=true", json={"points": pts}).raise_for_status()
        while s.get(base + "/collections/items").json()["result"]["status"] != "green":
            time.sleep(0.5)
        load_s = time.perf_counter() - t0
        local = threading.local()

        for label, params in [("Qdrant v1.12.4 (default HNSW)", None),
                              ("Qdrant v1.12.4 (hnsw_ef=256)", {"hnsw_ef": 256})]:
            def one(q, params=params):
                ss = getattr(local, "s", None)
                if ss is None:
                    ss = local.s = requests.Session()
                body = {"vector": q.tolist(), "limit": K}
                if params:
                    body["params"] = params
                r = ss.post(base + "/collections/items/points/search", json=body, timeout=600)
                r.raise_for_status()
                return [p["id"] for p in r.json()["result"]]

            found, lat, single, conc = run_queries(one, Q)
            out.append({"system": label, "recall@10": recall(found, truth),
                        "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), **qps(single, conc),
                        "load_s": load_s})
        steady = docker_mem(name)
        peak = sampler.stop()
        disk = int(subprocess.run(["docker", "exec", name, "du", "-sb", "/qdrant/storage"],
                                  capture_output=True, text=True).stdout.split()[0])
        for r in out:
            r.update(peak_mem_mib=peak / 2**20, steady_mem_mib=steady / 2**20, disk_mib=disk / 2**20)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = ["fashion", "glove"] if which == "all" else [which]
    path = HERE / "results" / "vectordb.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        X, Q, truth = load_dataset(name)
        print(f"== {name}: n={len(X):,} d={X.shape[1]} queries={len(Q)}", flush=True)
        rows = []
        for residency in ("memory", "shadow"):
            r = bench_heather(X, Q, truth, residency)
            print("  ", json.dumps(r), flush=True)
            rows.append(r)
        for r in bench_qdrant(X, Q, truth):
            print("  ", json.dumps(r), flush=True)
            rows.append(r)
        results[name] = {"n": len(X), "d": int(X.shape[1]), "queries": len(Q), "systems": rows,
                         "machine": {"cpu": "Apple M4", "cores": psutil.cpu_count(),
                                     "ram_gib": round(psutil.virtual_memory().total / 2**30)},
                         "heather_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                                          capture_output=True, text=True).stdout.strip(),
                         "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
