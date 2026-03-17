"""FastAPI backend for Cortex — a thinking notebook powered by HeatherDB."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import Cortex

app = FastAPI(title="Cortex API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")
cortex = Cortex(heather_url=HEATHER_URL)


# --- Request models ---

class WriteRequest(BaseModel):
    text: str

class LateralRequest(BaseModel):
    problem: str
    k: int = 5

class SurpriseRequest(BaseModel):
    text: str

class DreamRequest(BaseModel):
    hops: int = 5

class LoadRequest(BaseModel):
    texts: list[str]


# --- Routes ---

@app.post("/api/write")
def write_note(req: WriteRequest):
    if not req.text.strip():
        raise HTTPException(400, "Note text cannot be empty")
    result = cortex.write(req.text.strip())
    return result


@app.post("/api/lateral")
def lateral(req: LateralRequest):
    if not req.problem.strip():
        raise HTTPException(400, "Problem cannot be empty")
    result = cortex.lateral(req.problem.strip(), k=req.k)
    return asdict(result)


@app.get("/api/llm-status")
def llm_status():
    return {"enabled": cortex.llm_enabled()}


@app.get("/api/echo/{note_id}")
def echo(note_id: int):
    try:
        result = cortex.echo(note_id)
    except ValueError:
        raise HTTPException(404, "Note not found")
    return asdict(result)


@app.post("/api/surprise")
def surprise(req: SurpriseRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    result = cortex.surprise(req.text.strip())
    return {
        "novelty": result["novelty"],
        "familiar_notes": [asdict(n) for n in result["familiar_notes"]],
    }


@app.post("/api/dream")
def dream(req: DreamRequest):
    chain = cortex.dream(hops=req.hops)
    return asdict(chain)


@app.get("/api/landscape")
def landscape(probes: int = 20):
    basins = cortex.landscape(n_probes=probes)
    return {"basins": [asdict(b) for b in basins]}


@app.get("/api/notes")
def get_notes():
    notes = cortex.get_notes()
    return {"notes": notes}


@app.post("/api/load")
def load_notes(req: LoadRequest):
    if not req.texts:
        raise HTTPException(400, "No texts provided")
    count = cortex.load_notes(req.texts)
    return {"count": count}


@app.get("/api/stats")
def stats():
    return cortex.get_stats()


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    ok = cortex.delete_note(note_id)
    if not ok:
        raise HTTPException(404, "Note not found")
    return {"deleted": True}
