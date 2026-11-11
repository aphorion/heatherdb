#!/usr/bin/env python3
"""Head-to-head: one HeatherDB process vs the conventional multi-system stack.

    .venv/bin/python run.py                 # full suite -> results/results.json
    .venv/bin/python run.py --tasks fraud,classify

Accuracy: every task uses the same public dataset, split and metric for
HeatherDB and for the incumbents. Infrastructure: HeatherDB serves all six
capabilities from ONE server process, measured with every database loaded at
once; each incumbent runs in its own process (or container) and is measured
alone, as it would be deployed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import common

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TASKS = ["search", "discovery", "recs", "fraud", "classify", "access"]

# Incumbent runners: (task, label, module, function, loader-args)
BASELINES = {
    "discovery": [("run_kmeans_same_k", "k-means (HeatherDB's K)"),
                  ("run_kmeans_10", "k-means (true K=10, oracle)")],
    "recs": [("run_popularity", None), ("run_itemknn", None), ("run_als", None),
             ("run_itemknn_recency", None), ("run_als_recency", None)],
    "fraud": [("run_iforest", None), ("run_lof", None), ("run_lof50", None)],
    "classify": [("run_logreg", None), ("run_gbdt", None)],
    "access": [("run_logreg", None), ("run_gbdt", None)],
}


def child(task: str, fn: str, extra: str | None):
    """Run one incumbent in this (fresh) process and print its JSON."""
    mod = __import__(f"task_{task}")
    args = mod.load()[:-1]
    if task == "discovery":
        k = int(extra) if fn == "run_kmeans_same_k" else 10
        label = "k-means (HeatherDB's K)" if fn == "run_kmeans_same_k" else "k-means (true K=10, oracle)"
        res = mod.run_kmeans(*args, k, label)
    elif task == "recs":
        R, test_item, eval_users, _hist = args
        res = getattr(mod, fn)(R, test_item, eval_users)
    else:
        res = getattr(mod, fn)(*args)
    res["mem_bytes"] = common.peak_rss_self()
    res["process"] = "python service"
    print("@@RESULT@@" + json.dumps(res, default=float))


def run_child(task, fn, extra=None):
    cmd = [sys.executable, __file__, "--child", task, fn] + ([str(extra)] if extra else [])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    for line in out.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise RuntimeError(f"{task}.{fn} failed:\n{out.stderr[-2000:]}")


def qdrant_or_last(mod, args):
    """Qdrant is the incumbent: engine changes don't move its numbers. With
    Docker unavailable, reuse its last measured row, marked as carried over."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=True)
    except (subprocess.SubprocessError, OSError):
        prev = json.loads((RESULTS / "results.json").read_text())
        q = next(r for r in prev["tasks"]["search"]["systems"]
                 if r["system"].startswith("Qdrant"))
        q["carried_from"] = prev["started"]
        print(f"  Docker unavailable: Qdrant row carried from {prev['started']}", flush=True)
        return q
    q = mod.run_qdrant(*args)
    q["process"] = "docker container"
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--child", nargs="+")
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.child[2] if len(a.child) > 2 else None)
        return

    tasks = a.tasks.split(",")
    out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "machine": {"platform": platform.platform(), "cpu": platform.processor(),
                       "python": platform.python_version()},
           "heather": {"binary": str(common.BINARY),
                       "version": subprocess.run([str(common.BINARY), "--version"],
                                                 capture_output=True, text=True).stdout.strip(),
                       "commit": subprocess.run(["git", "describe", "--always", "--dirty"], cwd=HERE,
                                                capture_output=True, text=True).stdout.strip(),
                       "branch": subprocess.run(["git", "branch", "--show-current"], cwd=HERE,
                                                capture_output=True, text=True).stdout.strip()},
           "tasks": {}}
    h = common.Heather()
    try:
        for task in tasks:
            print(f"== {task}", flush=True)
            mod = __import__(f"task_{task}")
            loaded = mod.load()
            args, fp = loaded[:-1], loaded[-1]
            rec = {"dataset_fingerprint": fp, "doc": mod.__doc__, "systems": []}
            cpu0 = h.cpu_seconds()
            hres = mod.run_heather(h, *args)
            if isinstance(hres, tuple):  # recs returns (heather, raw-sum, freshness)
                hres, raw, fresh = hres
                rec["systems"].append(raw)
                rec["freshness"] = fresh
            hres["server_cpu_s"] = h.cpu_seconds() - cpu0
            # With selective attention (format 2) the engine's own result leads;
            # the same engine without it is kept as a comparison row.
            # Engine improvements, oldest first; the newest leads the task and
            # every earlier HeatherDB result stays as a comparison row.
            variants = [getattr(mod, f) for f in ("run_heather_sa", "run_heather_v2", "run_heather_sym")
                        if hasattr(mod, f)]
            if variants:
                hres["system"] = getattr(mod, "BEFORE_LABEL", "HeatherDB (no selective attention)")
                rows = [hres]
                for fn in variants:
                    cpu0 = h.cpu_seconds()
                    v = fn(h, *args)
                    v["server_cpu_s"] = h.cpu_seconds() - cpu0
                    rows.append(v)
                rec["systems"][0:0] = list(reversed(rows))
                hres = rows[-1]
            else:
                rec["systems"].insert(0, hres)
            print("  HeatherDB", {k: v for k, v in hres.items() if not isinstance(v, dict)}, flush=True)
            if task == "search":
                q = qdrant_or_last(mod, args)
                rec["systems"].append(q)
                print("  ", q, flush=True)
            for fn, _ in BASELINES.get(task, []):
                extra = hres["k"] if task == "discovery" else None
                r = run_child(task, fn, extra)
                rec["systems"].append(r)
                print("  ", r, flush=True)
            out["tasks"][task] = rec
            common.dump(out, RESULTS / "results.partial.json")
        # The convergence measurement: every capability loaded in one process.
        out["heather_footprint"] = {"peak_rss_bytes": h.peak_rss,
                                    "rss_all_loaded_bytes": h.sample(),
                                    "disk_bytes": h.disk_bytes(),
                                    "databases": h.call("GET", "/db")}
    finally:
        h.close()
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    common.dump(out, RESULTS / "results.json")
    print("wrote", RESULTS / "results.json")


if __name__ == "__main__":
    main()
