"""Core Cortex thinking engine — all intelligence comes from HeatherDB."""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from client import HeatherClient
from embeddings import Embedder, cosine_similarity, TARGET_DIMS

NOTES_DB = Path(__file__).parent / ".notes.db"


@dataclass
class NoteResult:
    id: int
    text: str
    similarity: float


@dataclass
class LateralInsight:
    source: NoteResult               # the lateral note from EAM
    bridge: str                      # LLM translation back to user's problem domain


@dataclass
class LateralResult:
    problem: str
    fidelity: float
    laterals: list[LateralInsight]   # SDM laterals + LLM bridges
    direct: list[NoteResult]         # direct matches (the obvious path)
    llm_answer: str | None           # Claude's "obvious" answer, if API key is set


@dataclass
class Echo:
    note: NoteResult
    fidelity: float                  # how faithfully SDM remembers this note
    interfering: list[NoteResult]    # other memories that bleed in during recall


@dataclass
class DreamHop:
    text: str
    similarity: float


@dataclass
class DreamChain:
    hops: list[DreamHop]
    theme: list[str]


@dataclass
class Basin:
    label: str
    fidelity: float
    notes: list[NoteResult]


class Cortex:
    def __init__(self, heather_url: str = "http://localhost:6380"):
        self.heather = HeatherClient(heather_url)
        self.embedder = Embedder()
        self._db = sqlite3.connect(str(NOTES_DB), check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._db.commit()

    def _store_note(self, text: str) -> int:
        cur = self._db.execute("INSERT INTO notes (text) VALUES (?)", (text,))
        self._db.commit()
        return cur.lastrowid

    def _get_all_notes(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, text, created_at FROM notes ORDER BY id"
        ).fetchall()
        return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    def _find_nearest_notes(
        self, vec: list[float], k: int = 5, exclude_ids: set[int] | None = None
    ) -> list[NoteResult]:
        notes = self._get_all_notes()
        if not notes:
            return []
        scored = []
        for note in notes:
            if exclude_ids and note["id"] in exclude_ids:
                continue
            note_vec = self.embedder.embed(note["text"])
            sim = cosine_similarity(vec, note_vec)
            scored.append(NoteResult(id=note["id"], text=note["text"], similarity=sim))
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:k]

    def write(self, text: str) -> dict:
        vec = self.embedder.embed(text)
        self.heather.write([vec])
        note_id = self._store_note(text)
        # Compute fidelity hint: how well does SDM already know this?
        reconstructed = self.heather.read(vec, strategy="iterative")
        fidelity = cosine_similarity(vec, reconstructed)
        return {"id": note_id, "fidelity_hint": round(fidelity, 4)}

    def lateral(self, problem: str, k: int = 5) -> LateralResult:
        vec = self.embedder.embed(problem)

        # EAM reconstruction — the database recalls what it "knows"
        reconstructed = self.heather.read(vec, strategy="iterative")
        fidelity = cosine_similarity(vec, reconstructed)

        # Notes nearest to EAM reconstruction (wider net for laterals)
        eam_nearest = self._find_nearest_notes(reconstructed, k=k + 3)

        # Notes nearest to the raw query (the obvious matches)
        direct = self._find_nearest_notes(vec, k=k)

        # Laterals: notes near reconstruction but NOT near query
        direct_ids = {r.id for r in direct}
        raw_laterals = [r for r in eam_nearest if r.id not in direct_ids][:k]

        # Build bridges: LLM translates each lateral back to the user's domain
        laterals = []
        for lat in raw_laterals:
            bridge = self._build_bridge(problem, lat.text)
            laterals.append(LateralInsight(source=lat, bridge=bridge or ""))

        # Optional LLM "obvious" answer for contrast
        llm_answer = self._call_llm(problem)

        return LateralResult(
            problem=problem,
            fidelity=round(fidelity, 4),
            laterals=laterals,
            direct=direct,
            llm_answer=llm_answer,
        )

    def llm_enabled(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _call_llm(self, problem: str) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                system="You are a concise problem-solving assistant. Give a practical, conventional answer in 2-3 sentences. Stick to well-known, standard approaches. Do not be creative or lateral — give the textbook answer.",
                messages=[{"role": "user", "content": problem}],
            )
            return resp.content[0].text
        except Exception:
            return None

    def _build_bridge(self, problem: str, lateral_note: str) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=200,
                system=(
                    "You are a lateral thinking translator. The user has a problem, "
                    "and a memory system found a structural analog from a completely "
                    "different domain. Your job: explain in 1-2 sentences how the "
                    "analog applies to their specific problem. Be concrete and "
                    "actionable. Start with the structural insight, then give a "
                    "specific action. Do NOT repeat the analog — go straight to "
                    "the application."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Problem: {problem}\n\n"
                        f"Structural analog from another domain: {lateral_note}\n\n"
                        f"How does this analog apply to the problem? Be specific and actionable."
                    ),
                }],
            )
            return resp.content[0].text
        except Exception:
            return None

    def echo(self, note_id: int) -> Echo:
        row = self._db.execute(
            "SELECT id, text FROM notes WHERE id=?", (note_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Note {note_id} not found")

        note = NoteResult(id=row[0], text=row[1], similarity=1.0)
        note_vec = self.embedder.embed(note.text)

        # Read through SDM — how does the database remember this note?
        reconstructed = self.heather.read(note_vec, strategy="iterative")
        fidelity = cosine_similarity(note_vec, reconstructed)

        # What other memories bleed into this one during recall?
        interfering = self._find_nearest_notes(
            reconstructed, k=5, exclude_ids={note_id}
        )

        return Echo(
            note=note,
            fidelity=round(fidelity, 4),
            interfering=interfering,
        )

    def surprise(self, text: str) -> dict:
        vec = self.embedder.embed(text)
        reconstructed = self.heather.read(vec, strategy="iterative")
        fidelity = cosine_similarity(vec, reconstructed)

        # Novelty is the inverse of fidelity
        novelty = round(1.0 - fidelity, 4)

        # Find what it reminds the database of — use the raw vector,
        # not the reconstruction (which drifts to dominant attractors)
        familiar = self._find_nearest_notes(vec, k=5)

        return {"novelty": novelty, "familiar_notes": familiar}

    def dream(self, hops: int = 5) -> DreamChain:
        # Start from random noise
        rng = np.random.default_rng()
        current = rng.normal(0, 1, TARGET_DIMS).tolist()
        norm = np.linalg.norm(current)
        if norm > 0:
            current = (np.array(current) / norm).tolist()

        chain = []
        seen_ids = set()

        for _ in range(hops):
            reconstructed = self.heather.read(current, strategy="iterative")
            nearest = self._find_nearest_notes(reconstructed, k=3, exclude_ids=seen_ids)
            if not nearest:
                break
            top = nearest[0]
            chain.append(DreamHop(text=top.text, similarity=top.similarity))
            seen_ids.add(top.id)
            # Next hop: use the found note's embedding as next query
            current = self.embedder.embed(top.text)

        # Extract theme: common words across hops
        theme = self._extract_theme(chain)

        return DreamChain(hops=chain, theme=theme)

    def landscape(self, n_probes: int = 20) -> list[Basin]:
        rng = np.random.default_rng()
        basins: dict[int, list[NoteResult]] = {}
        basin_vecs: dict[int, list[float]] = {}

        for _ in range(n_probes):
            probe = rng.normal(0, 1, TARGET_DIMS).tolist()
            norm = np.linalg.norm(probe)
            if norm > 0:
                probe = (np.array(probe) / norm).tolist()

            reconstructed = self.heather.read(probe, strategy="iterative")
            nearest = self._find_nearest_notes(reconstructed, k=3)
            if not nearest:
                continue

            # Assign to basin by top note id
            top_id = nearest[0].id
            if top_id not in basins:
                basins[top_id] = []
                basin_vecs[top_id] = reconstructed
            # Merge notes
            seen = {n.id for n in basins[top_id]}
            for n in nearest:
                if n.id not in seen:
                    basins[top_id].append(n)
                    seen.add(n.id)

        # Convert to Basin objects
        result = []
        for top_id, notes in basins.items():
            # Fidelity = how strongly this basin attracts
            vec = basin_vecs[top_id]
            re_read = self.heather.read(vec, strategy="iterative")
            fidelity = cosine_similarity(vec, re_read)

            # Label from common words
            all_text = " ".join(n.text for n in notes)
            words = all_text.lower().split()
            stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                     "at", "to", "for", "of", "and", "or", "but", "with", "by",
                     "from", "as", "it", "its", "this", "that", "can", "be",
                     "has", "have", "had", "do", "does", "did", "will", "would",
                     "when", "where", "how", "what", "which", "who", "their",
                     "they", "them", "not", "no", "if", "than", "then", "so",
                     "just", "also", "like", "into", "over", "such", "through",
                     "between", "each", "other", "more", "most", "about", "up",
                     "out", "new", "one", "two", "may", "could", "some", "all",
                     "these", "those", "only", "many", "much", "same", "own",
                     "even", "both", "few", "well", "very", "often"}
            freq = {}
            for w in words:
                w = w.strip(".,!?;:()\"'")
                if len(w) > 2 and w not in stop:
                    freq[w] = freq.get(w, 0) + 1
            top_words = sorted(freq, key=freq.get, reverse=True)[:3]
            label = " / ".join(top_words) if top_words else "unknown"

            result.append(Basin(
                label=label,
                fidelity=round(fidelity, 4),
                notes=notes[:5],
            ))

        result.sort(key=lambda b: b.fidelity, reverse=True)
        return result

    def get_notes(self) -> list[dict]:
        return self._get_all_notes()

    def load_notes(self, texts: list[str]) -> int:
        vecs = self.embedder.embed_batch(texts)
        count = 0
        for text, vec in zip(texts, vecs):
            self.heather.write([vec])
            self._store_note(text)
            count += 1
        return count

    def get_stats(self) -> dict:
        notes = self._get_all_notes()
        heather_stats = self.heather.stats()
        return {"note_count": len(notes), "heather_stats": heather_stats}

    def delete_note(self, note_id: int) -> bool:
        cur = self._db.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self._db.commit()
        return cur.rowcount > 0

    def _extract_theme(self, chain: list[DreamHop]) -> list[str]:
        if not chain:
            return []
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "and", "or", "but", "with", "by",
                 "from", "as", "it", "its", "this", "that", "can", "be"}
        word_counts: dict[str, int] = {}
        for hop in chain:
            words = set()
            for w in hop.text.lower().split():
                w = w.strip(".,!?;:()\"'")
                if len(w) > 2 and w not in stop:
                    words.add(w)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Words appearing in multiple hops
        theme = [w for w, c in word_counts.items() if c >= 2]
        theme.sort(key=lambda w: word_counts[w], reverse=True)
        return theme[:5]
