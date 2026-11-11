"""Shared plumbing: the HeatherDB process, its HTTP client, and resource meters.

HeatherDB is driven over its public HTTP API with authentication ON
(Basic once, then a Bearer session token), exactly as an application would
use it. Baselines run in-process inside a child Python process so that each
one's peak memory is measured in isolation.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import requests

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("HEATHER_BIN", ROOT / "target/release/heather"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def dir_bytes(path: Path) -> int:
    """Allocated bytes on disk (LMDB files are sparse; st_size overstates)."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_blocks * 512
    return total


class Heather:
    """One HeatherDB server process: the single foundation every task shares."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or tempfile.mkdtemp(prefix="h2h-heather-"))
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ,
                   HEATHER_DATA_DIR=str(self.data_dir),
                   HEATHER_PORT=str(self.port),
                   HEATHER_HOST="127.0.0.1",
                   HEATHER_ADMIN_USER="bench",
                   HEATHER_ADMIN_PASSWORD="bench-password-1234",
                   HEATHER_MAX_BODY_SIZE=str(64 * 1024 * 1024),
                   HEATHER_REQUEST_TIMEOUT="600",
                   RUST_LOG="warn")
        env.pop("HEATHER_AUTH_DISABLED", None)
        self.proc = subprocess.Popen([str(BINARY)], env=env,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self.ps = psutil.Process(self.proc.pid)
        self.s = requests.Session()
        for _ in range(200):
            try:
                if self.s.get(self.base + "/health", timeout=1).ok:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.05)
        else:
            raise RuntimeError("heather did not start")
        tok = self.s.post(self.base + "/auth/token",
                          auth=("bench", "bench-password-1234")).json()["token"]
        self.s.headers["Authorization"] = f"Bearer {tok}"
        self.peak_rss = 0

    # -- HTTP --------------------------------------------------------------
    def call(self, method: str, path: str, body=None):
        r = self.s.request(method, self.base + path, json=body, timeout=900)
        self.sample()
        if not r.ok:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def create_db(self, name: str, dim: int, mdl_gate: bool = False, **eam):
        body = {"name": name, "dimension": dim, "map_size_mb": 65536,
                "mdl_gate": mdl_gate}
        if eam:
            body["eam"] = eam
        out = self.call("POST", "/db", body)
        # Bulk import, as Postgres COPY with synchronous_commit=off: commits
        # reach the OS, not the disk, one at a time. The incumbents load into
        # memory with no per-write sync either. Reads are unaffected.
        self.call("PUT", f"/db/{name}/durability", {"sync": False})
        return out

    def create_columns_db(self, name: str, columns: dict, predict=(), sequence=False):
        """A database declared by its columns; rows are then written and read
        by name (`rows`, `predict`, `next?column=`). Bulk import, as above."""
        out = self.call("POST", "/db", {"name": name, "map_size_mb": 65536, "columns": columns,
                                        "predict": list(predict), "sequence": sequence})
        self.call("PUT", f"/db/{name}/durability", {"sync": False})
        return out

    # -- measurement -------------------------------------------------------
    def sample(self) -> int:
        try:
            rss = self.ps.memory_info().rss
        except psutil.Error:
            return self.peak_rss
        self.peak_rss = max(self.peak_rss, rss)
        return rss

    def cpu_seconds(self) -> float:
        t = self.ps.cpu_times()
        return t.user + t.system

    def disk_bytes(self) -> int:
        return dir_bytes(self.data_dir)

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.data_dir, ignore_errors=True)


def b64rows(X: np.ndarray) -> list[str]:
    X = np.ascontiguousarray(X, dtype="<f8")
    return [base64.b64encode(row.tobytes()).decode() for row in X]


def unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


class Meter:
    """Wall time and CPU seconds of *this* process for a block."""

    def __enter__(self):
        self.p = psutil.Process()
        t = self.p.cpu_times()
        self.cpu0 = t.user + t.system
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        t = self.p.cpu_times()
        self.cpu = t.user + t.system - self.cpu0


def peak_rss_self() -> int:
    """Peak RSS of this process in bytes (macOS reports bytes, Linux KiB)."""
    import resource
    import sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def pct(xs, q):
    return float(np.percentile(np.asarray(xs) * 1000.0, q))  # -> ms


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
