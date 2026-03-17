"""Recall API — SDM-powered smart context compression for LLM conversations."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

from engine import RecallEngine

HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")
DB_PATH = str(Path(__file__).parent / ".recall.db")

app = FastAPI(title="Recall")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RecallEngine(heather_url=HEATHER_URL, db_path=DB_PATH)


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    result = engine.chat(req.message.strip())

    return {
        "response": result.response,
        "recalled_memories": [
            {
                "id": m.message.id,
                "role": m.message.role,
                "content": m.message.content,
                "similarity": m.similarity,
                "memory_type": m.memory_type,
                "timestamp": m.message.timestamp,
            }
            for m in result.recalled_memories
        ],
        "fidelity": round(result.fidelity, 4),
        "recent_tokens": result.recent_tokens,
        "recalled_tokens": result.recalled_tokens,
    }


@app.get("/api/messages")
def get_messages(limit: int = 50):
    messages = engine.get_messages(limit)
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in messages
        ]
    }


@app.get("/api/stats")
def stats():
    return engine.get_stats()


@app.delete("/api/reset")
def reset():
    """Clear all messages (for testing)."""
    engine._db.execute("DELETE FROM message_embeddings")
    engine._db.execute("DELETE FROM messages")
    engine._db.commit()
    return {"status": "reset"}


@app.get("/api/health")
def health():
    heather_ok = engine.heather.health()
    return {"status": "ok", "heather_ok": heather_ok}
