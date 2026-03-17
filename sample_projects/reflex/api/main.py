"""Reflex — Visual pattern completion powered by SDM. No ML, no embeddings.

An 8x8 grid of cells. Each pattern is a raw 64-dimensional vector.
The EAM learns patterns and completes partial ones through memory interference.
"""

from __future__ import annotations

import os
import sqlite3
import json
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Reflex API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRID_SIZE = 8
DIMS = GRID_SIZE * GRID_SIZE  # 64
HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")
DB_PATH = Path(__file__).parent / ".reflex.db"

# --- HeatherDB client ---

_http = httpx.Client(base_url=HEATHER_URL, timeout=30.0)


def heather_write(vec: list[float]) -> None:
    _http.post("/write", json={"vectors": [vec]}).raise_for_status()


def heather_read(vec: list[float], strategy: str = "iterative") -> list[float]:
    resp = _http.post("/read", json={"query": vec, "strategy": strategy})
    resp.raise_for_status()
    return resp.json()["result"]


def heather_stats() -> dict:
    return _http.get("/stats").json()


# --- SQLite for pattern metadata ---

_db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_db.execute(
    """CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        grid TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
)
_db.commit()


# --- Encoding ---

def grid_to_vec(grid: list[list[int]]) -> list[float]:
    """Flatten 8x8 grid to 64-dim unit vector. Cell values: 0=empty, 1=filled."""
    flat = []
    for row in grid:
        for cell in row:
            flat.append(float(cell))
    vec = np.array(flat, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def vec_to_grid(vec: list[float]) -> list[list[float]]:
    """Reshape 64-dim vector back to 8x8 grid of continuous values."""
    arr = np.array(vec[:DIMS], dtype=np.float64)
    return arr.reshape(GRID_SIZE, GRID_SIZE).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# --- Request models ---

class WriteRequest(BaseModel):
    grid: list[list[int]]
    name: str = ""

class CompleteRequest(BaseModel):
    grid: list[list[int]]

class BlendRequest(BaseModel):
    pattern_ids: list[int]

class SurpriseRequest(BaseModel):
    grid: list[list[int]]


# --- Routes ---

@app.post("/api/write")
def write_pattern(req: WriteRequest):
    if len(req.grid) != GRID_SIZE or any(len(r) != GRID_SIZE for r in req.grid):
        raise HTTPException(400, f"Grid must be {GRID_SIZE}x{GRID_SIZE}")
    vec = grid_to_vec(req.grid)
    heather_write(vec)
    cur = _db.execute(
        "INSERT INTO patterns (name, grid) VALUES (?, ?)",
        (req.name, json.dumps(req.grid)),
    )
    _db.commit()
    # Fidelity: how well does SDM already know this?
    reconstructed = heather_read(vec)
    fidelity = cosine_similarity(vec, reconstructed)
    return {"id": cur.lastrowid, "fidelity": round(fidelity, 4)}


@app.post("/api/complete")
def complete_pattern(req: CompleteRequest):
    if len(req.grid) != GRID_SIZE or any(len(r) != GRID_SIZE for r in req.grid):
        raise HTTPException(400, f"Grid must be {GRID_SIZE}x{GRID_SIZE}")

    query_vec = grid_to_vec(req.grid)
    reconstructed = heather_read(query_vec)
    fidelity = cosine_similarity(query_vec, reconstructed)

    # Convert reconstruction back to grid
    recon_grid = vec_to_grid(reconstructed)

    # Determine which cells the EAM "completed" (high value in reconstruction
    # where the user left empty)
    completed = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    confidence = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

    # Find max value for normalization
    flat_recon = [recon_grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    max_val = max(abs(v) for v in flat_recon) if flat_recon else 1.0
    if max_val == 0:
        max_val = 1.0

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            norm_val = recon_grid[r][c] / max_val
            confidence[r][c] = round(max(0.0, norm_val), 4)
            if req.grid[r][c] == 0 and norm_val > 0.3:
                completed[r][c] = 1

    return {
        "completed": completed,
        "confidence": confidence,
        "fidelity": round(fidelity, 4),
        "original": req.grid,
    }


@app.post("/api/blend")
def blend_patterns(req: BlendRequest):
    if len(req.pattern_ids) < 2:
        raise HTTPException(400, "Need at least 2 patterns")

    vecs = []
    for pid in req.pattern_ids:
        row = _db.execute("SELECT grid FROM patterns WHERE id=?", (pid,)).fetchone()
        if not row:
            raise HTTPException(404, f"Pattern {pid} not found")
        grid = json.loads(row[0])
        vecs.append(grid_to_vec(grid))

    # Average the vectors
    blend = np.mean(vecs, axis=0)
    norm = np.linalg.norm(blend)
    if norm > 0:
        blend = blend / norm

    # Read through SDM
    reconstructed = heather_read(blend.tolist())
    fidelity = cosine_similarity(blend.tolist(), reconstructed)
    recon_grid = vec_to_grid(reconstructed)

    # Normalize to binary
    flat_recon = [recon_grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    max_val = max(abs(v) for v in flat_recon) if flat_recon else 1.0
    if max_val == 0:
        max_val = 1.0

    result = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    confidence = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            norm_val = recon_grid[r][c] / max_val
            confidence[r][c] = round(max(0.0, norm_val), 4)
            if norm_val > 0.3:
                result[r][c] = 1

    return {
        "grid": result,
        "confidence": confidence,
        "fidelity": round(fidelity, 4),
    }


@app.post("/api/surprise")
def surprise(req: SurpriseRequest):
    if len(req.grid) != GRID_SIZE or any(len(r) != GRID_SIZE for r in req.grid):
        raise HTTPException(400, f"Grid must be {GRID_SIZE}x{GRID_SIZE}")
    vec = grid_to_vec(req.grid)
    reconstructed = heather_read(vec)
    fidelity = cosine_similarity(vec, reconstructed)
    return {"novelty": round(1.0 - fidelity, 4), "fidelity": round(fidelity, 4)}


@app.get("/api/patterns")
def list_patterns():
    rows = _db.execute(
        "SELECT id, name, grid, created_at FROM patterns ORDER BY id"
    ).fetchall()
    return {
        "patterns": [
            {"id": r[0], "name": r[1], "grid": json.loads(r[2]), "created_at": r[3]}
            for r in rows
        ]
    }


@app.delete("/api/patterns/{pattern_id}")
def delete_pattern(pattern_id: int):
    cur = _db.execute("DELETE FROM patterns WHERE id=?", (pattern_id,))
    _db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Pattern not found")
    return {"deleted": True}


@app.get("/api/stats")
def stats():
    count = _db.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    return {"pattern_count": count, "heather_stats": heather_stats()}
