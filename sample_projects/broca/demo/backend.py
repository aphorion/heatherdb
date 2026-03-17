#!/usr/bin/env python3
"""HeatherDB Knowledge Demo — FastAPI backend.

Connects to HeatherDB, loads frozen encoder/decoder + adapter,
exposes generation and knowledge algebra endpoints.

Usage:
    pip install fastapi uvicorn httpx sentence-transformers transformers torch
    python backend.py
"""

import os
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", str(Path(__file__).parent.parent / "broca_v2_adapter.pt"))
DEVICE = "cpu"  # No GPU — that's the point

THOUGHT_DIM = 384
GPT2_DIM = 768
N_PREFIX = 8

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ThoughtAdapter(nn.Module):
    def __init__(self, thought_dim=384, gpt2_dim=768, n_prefix=8):
        super().__init__()
        self.n_prefix = n_prefix
        self.gpt2_dim = gpt2_dim
        self.project = nn.Sequential(
            nn.Linear(thought_dim, 128),
            nn.GELU(),
            nn.Linear(128, gpt2_dim * n_prefix),
        )

    def forward(self, thought):
        return self.project(thought).view(-1, self.n_prefix, self.gpt2_dim)


# Global model state
encoder = None
gpt2 = None
tokenizer = None
adapter = None
heather = None


def load_models():
    global encoder, gpt2, tokenizer, adapter, heather
    from sentence_transformers import SentenceTransformer
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading encoder (sentence-transformers)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(False)

    print("Loading decoder (GPT-2)...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
    for p in gpt2.parameters():
        p.requires_grad_(False)
    gpt2.eval()

    print("Loading adapter...")
    adapter = ThoughtAdapter(THOUGHT_DIM, GPT2_DIM, N_PREFIX).to(DEVICE)
    if Path(ADAPTER_PATH).exists():
        adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=DEVICE, weights_only=True))
        print(f"  Loaded from {ADAPTER_PATH}")
    else:
        print(f"  WARNING: {ADAPTER_PATH} not found, using random adapter")
    adapter.eval()

    heather = httpx.Client(base_url=HEATHER_URL, timeout=30)
    print(f"HeatherDB: {HEATHER_URL}")


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(prompt: str, collection: str, temperature: float = 0.8,
                  max_tokens: int = 150) -> dict:
    """Full pipeline: encode → HeatherDB read → adapter → GPT-2 generate."""
    timings = {}

    # 1. Encode prompt
    t0 = time.perf_counter()
    thought = encoder.encode(prompt, convert_to_tensor=True, device=DEVICE)
    if thought.dim() == 1:
        thought = thought.unsqueeze(0)
    timings["encode_ms"] = (time.perf_counter() - t0) * 1000

    # 2. HeatherDB read (disk)
    t1 = time.perf_counter()
    query = thought[0].numpy().astype(np.float64).tolist()
    resp = heather.post(
        f"/collections/{collection}/read",
        json={"query": query, "strategy": "iterative"},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HeatherDB read failed: {resp.text}")

    result = np.array(resp.json()["result"], dtype=np.float32)
    thought_enriched = torch.tensor(result, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    timings["disk_ms"] = (time.perf_counter() - t1) * 1000

    # Similarity between raw and enriched thought
    sim = F.cosine_similarity(thought, thought_enriched).item()

    # 3. Adapter + GPT-2 decode
    t2 = time.perf_counter()
    prefix = adapter(thought_enriched)

    outputs = gpt2(inputs_embeds=prefix, use_cache=True)
    past = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :] / temperature

    generated_ids = []
    for _ in range(max_tokens):
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated_ids.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        outputs = gpt2(input_ids=next_token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :] / temperature

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    timings["decode_ms"] = (time.perf_counter() - t2) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "text": text,
        "collection": collection,
        "similarity": round(sim, 4),
        "timings": {k: round(v, 1) for k, v in timings.items()},
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    if heather:
        heather.close()

app = FastAPI(title="HeatherDB Knowledge Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class GenerateRequest(BaseModel):
    prompt: str
    collection: str
    temperature: float = 0.8
    max_tokens: int = 150

class CompareRequest(BaseModel):
    prompt: str
    collections: list[str]
    temperature: float = 0.8
    max_tokens: int = 150

class AlgebraRequest(BaseModel):
    operation: str  # "add", "sub", "scale"
    source_a: str
    source_b: str | None = None
    target: str
    alpha: float | None = None
    max_cross_k: int = 5

class AnalyzeRequest(BaseModel):
    prompt: str
    collection: str


# --- Endpoints ---

@app.get("/api/collections")
def list_collections():
    resp = heather.get("/collections")
    collections = resp.json().get("collections", [])
    results = []
    for name in collections:
        info = {"name": name}
        try:
            fp_resp = heather.get(f"/collections/{name}/fingerprint")
            fp = fp_resp.json().get("fingerprint")
            info["has_fingerprint"] = fp is not None
        except Exception:
            info["has_fingerprint"] = False
        results.append(info)
    return {"collections": results}


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    return generate_text(req.prompt, req.collection, req.temperature, req.max_tokens)


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    results = []
    for coll in req.collections:
        try:
            r = generate_text(req.prompt, coll, req.temperature, req.max_tokens)
            results.append(r)
        except Exception as e:
            results.append({"collection": coll, "error": str(e)})
    return {"results": results}


@app.post("/api/algebra")
def api_algebra(req: AlgebraRequest):
    if req.operation == "add":
        resp = heather.post("/algebra/add", json={
            "source_a": req.source_a,
            "source_b": req.source_b,
            "target": req.target,
            "max_cross_k": req.max_cross_k,
        })
    elif req.operation == "sub":
        resp = heather.post("/algebra/sub", json={
            "source_a": req.source_a,
            "source_b": req.source_b,
            "target": req.target,
            "max_cross_k": req.max_cross_k,
        })
    elif req.operation == "scale":
        resp = heather.post("/algebra/scale", json={
            "source": req.source_a,
            "target": req.target,
            "alpha": req.alpha or 1.0,
        })
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {req.operation}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Algebra failed: {resp.text}")

    return resp.json()


@app.get("/api/collection/{name}/stats")
def collection_stats(name: str):
    fp_resp = heather.get(f"/collections/{name}/fingerprint")
    fp_data = fp_resp.json()
    return {
        "name": name,
        "fingerprint": fp_data.get("fingerprint") is not None,
    }


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    thought = encoder.encode(req.prompt, convert_to_tensor=True, device=DEVICE)
    if thought.dim() == 1:
        thought = thought.unsqueeze(0)
    query = thought[0].numpy().astype(np.float64).tolist()

    resp = heather.post(
        f"/collections/{req.collection}/analyze",
        json={"query": query, "strategy": "iterative"},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Analyze failed: {resp.text}")

    data = resp.json()
    data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return data


# --- Static files ---

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(str(static_dir / "index.html"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
