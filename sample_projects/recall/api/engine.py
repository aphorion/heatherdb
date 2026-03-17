"""
Recall Engine — SDM as long-term associative memory for LLM conversations.

Context window = working memory (recent messages, fixed budget).
SDM = long-term memory (all past messages, retrieved by associative recall).

On each user message:
  1. Embed it
  2. Query SDM — reconstruction blends all similar past exchanges
  3. Find stored messages nearest to the RECONSTRUCTION (not the query)
  4. Build prompt: system + recalled context + recent messages + current
  5. Call Claude
  6. Write exchange to SDM + SQLite
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic

from client import HeatherClient
from embeddings import Embedder, cosine_similarity


# ── data models ─────────────────────────────────────────────────────

@dataclass
class Message:
    id: int
    role: str
    content: str
    timestamp: str


@dataclass
class RecalledMemory:
    message: Message
    similarity: float
    memory_type: str  # "vivid" | "clear" | "vague"


@dataclass
class ChatResult:
    response: str
    recalled_memories: list[RecalledMemory]
    fidelity: float
    recent_tokens: int
    recalled_tokens: int


# ── token helpers ───────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token for Claude)."""
    return len(text) // 4


# ── engine ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with long-term associative memory.

When <recalled_context> is present, it contains memories from past conversations that were
associatively reconstructed based on the user's current message. Each memory is labeled:
- [vivid] — strong match, highly relevant
- [clear] — moderate match, likely relevant
- [vague] — weak match, use cautiously

The fidelity score indicates how strongly the memory system recognizes the current topic.
High fidelity means these memories are reliable. Low fidelity means this is likely a new topic.

Use recalled memories to maintain continuity across the full conversation history.
Prioritize recent messages for immediate context. Reference recalled memories naturally
when they're relevant — don't announce that you're using memory unless asked."""


class RecallEngine:
    TOTAL_BUDGET = 8000
    SYSTEM_TOKENS = 300
    SDM_BASE = 2000
    RECENT_BASE = 4000

    def __init__(self, heather_url: str, db_path: str):
        self.heather = HeatherClient(heather_url)
        self.embedder = Embedder()
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self._db_path = db_path
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        # cache embeddings alongside messages
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                embedding TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        """)
        self._db.commit()

    # ── core chat flow ──────────────────────────────────────────────

    def chat(self, user_message: str) -> ChatResult:
        # 1. embed user message
        user_vec = self.embedder.embed(user_message)

        # 2. query SDM for associative reconstruction
        try:
            reconstructed = self.heather.read(user_vec, strategy="iterative")
            fidelity = cosine_similarity(user_vec, reconstructed)
        except Exception:
            reconstructed = user_vec
            fidelity = 0.0

        # 3. find stored messages nearest to the RECONSTRUCTION
        recalled = self._recall_memories(reconstructed, fidelity)

        # 4. adaptive token budgets
        eam_budget = self.SDM_BASE + int((fidelity - 0.5) * 2000)
        eam_budget = max(500, min(3000, eam_budget))
        recent_budget = self.TOTAL_BUDGET - self.SYSTEM_TOKENS - eam_budget

        # 5. build recalled context string
        recalled_context, recalled_tokens = self._format_recalled(
            recalled, eam_budget, fidelity
        )

        # 6. get recent messages within budget
        recent_msgs = self._get_recent_messages(recent_budget)
        recent_tokens = sum(count_tokens(m.content) + 10 for m in recent_msgs)

        # 7. build prompt
        system, messages = self._build_prompt(
            recalled_context, recent_msgs, user_message
        )

        # 8. call Claude
        response_text = self._call_claude(system, messages)

        # 9. store exchange
        self._store_exchange(user_message, response_text, user_vec)

        return ChatResult(
            response=response_text,
            recalled_memories=recalled,
            fidelity=fidelity,
            recent_tokens=recent_tokens,
            recalled_tokens=recalled_tokens,
        )

    # ── memory retrieval ────────────────────────────────────────────

    def _recall_memories(
        self, reconstructed: list[float], fidelity: float
    ) -> list[RecalledMemory]:
        """Find messages nearest to EAM reconstruction."""
        rows = self._db.execute(
            "SELECT m.id, m.role, m.content, m.timestamp, e.embedding "
            "FROM messages m JOIN message_embeddings e ON m.id = e.message_id"
        ).fetchall()

        if not rows:
            return []

        import json
        scored: list[RecalledMemory] = []
        for row in rows:
            msg = Message(id=row[0], role=row[1], content=row[2], timestamp=row[3])
            msg_vec = json.loads(row[4])
            sim = cosine_similarity(reconstructed, msg_vec)

            if sim > 0.50:
                if sim > 0.85:
                    mtype = "vivid"
                elif sim > 0.70:
                    mtype = "clear"
                else:
                    mtype = "vague"
                scored.append(RecalledMemory(msg, round(sim, 4), mtype))

        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:10]

    def _format_recalled(
        self, recalled: list[RecalledMemory], budget: int, fidelity: float
    ) -> tuple[str, int]:
        """Format recalled memories into context string within token budget."""
        if not recalled:
            return "", 0

        lines = [f'<recalled_context fidelity="{fidelity:.2f}">']
        tokens_used = count_tokens(lines[0]) + 5  # closing tag

        for mem in recalled:
            entry = (
                f"[{mem.memory_type} memory {int(mem.similarity * 100)}%] "
                f"{mem.message.role}: {mem.message.content}"
            )
            entry_tokens = count_tokens(entry)
            if tokens_used + entry_tokens > budget:
                break
            lines.append(entry)
            tokens_used += entry_tokens

        lines.append("</recalled_context>")

        if len(lines) <= 2:  # only tags, no memories fit
            return "", 0

        return "\n".join(lines), tokens_used

    # ── recent messages ─────────────────────────────────────────────

    def _get_recent_messages(self, token_budget: int) -> list[Message]:
        rows = self._db.execute(
            "SELECT id, role, content, timestamp FROM messages ORDER BY id DESC"
        ).fetchall()

        recent: list[Message] = []
        used = 0
        for row in rows:
            msg = Message(id=row[0], role=row[1], content=row[2], timestamp=row[3])
            msg_tokens = count_tokens(msg.content) + 10
            if used + msg_tokens > token_budget:
                break
            recent.insert(0, msg)
            used += msg_tokens

        return recent

    # ── prompt building ─────────────────────────────────────────────

    def _build_prompt(
        self,
        recalled_context: str,
        recent: list[Message],
        user_message: str,
    ) -> tuple[str, list[dict]]:
        messages = []

        # inject recalled context as early context
        if recalled_context:
            messages.append({"role": "user", "content": recalled_context})
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the recalled context from our conversation history."
            })

        # add recent messages
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})

        # add current user message
        messages.append({"role": "user", "content": user_message})

        return SYSTEM_PROMPT, messages

    # ── Claude API ──────────────────────────────────────────────────

    def _call_claude(self, system: str, messages: list[dict]) -> str:
        resp = self.anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    # ── storage ─────────────────────────────────────────────────────

    def _store_exchange(
        self, user_msg: str, assistant_msg: str, user_vec: list[float]
    ):
        import json

        now = datetime.now().isoformat()

        # store user message
        cursor = self._db.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            ("user", user_msg, now),
        )
        user_id = cursor.lastrowid
        self._db.execute(
            "INSERT INTO message_embeddings (message_id, embedding) VALUES (?, ?)",
            (user_id, json.dumps(user_vec)),
        )

        # store assistant message
        assistant_vec = self.embedder.embed(assistant_msg)
        cursor = self._db.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            ("assistant", assistant_msg, now),
        )
        assistant_id = cursor.lastrowid
        self._db.execute(
            "INSERT INTO message_embeddings (message_id, embedding) VALUES (?, ?)",
            (assistant_id, json.dumps(assistant_vec)),
        )

        self._db.commit()

        # write to SDM: user vec, assistant vec, combined exchange vec
        exchange_vec = self.embedder.embed(
            f"User: {user_msg}\nAssistant: {assistant_msg}"
        )
        try:
            self.heather.write([user_vec, assistant_vec, exchange_vec])
        except Exception:
            pass  # SDM might be down, conversation still works via SQLite

    # ── stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        msg_count = self._db.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        try:
            heather_stats = self.heather.stats()
        except Exception:
            heather_stats = {}
        return {
            "message_count": msg_count,
            "heather_stats": heather_stats,
        }

    def get_messages(self, limit: int = 50) -> list[Message]:
        rows = self._db.execute(
            "SELECT id, role, content, timestamp FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Message(id=r[0], role=r[1], content=r[2], timestamp=r[3])
            for r in reversed(rows)
        ]
