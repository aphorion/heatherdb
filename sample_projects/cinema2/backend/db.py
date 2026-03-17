"""SQLite persistence for Cinema2 — catalog, profiles, OMDb cache."""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "cinema2.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS movies (
            imdb_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            year TEXT,
            rated TEXT,
            runtime TEXT,
            genres TEXT,       -- JSON array
            director TEXT,
            actors TEXT,       -- JSON array
            plot TEXT,
            poster TEXT,
            imdb_rating TEXT,
            imdb_votes TEXT
        );

        CREATE TABLE IF NOT EXISTS profiles (
            name TEXT PRIMARY KEY,
            fingerprint TEXT,  -- JSON array (floats)
            fidelity REAL,
            evolution TEXT     -- JSON array of steps
        );

        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_name TEXT NOT NULL REFERENCES profiles(name) ON DELETE CASCADE,
            imdb_id TEXT NOT NULL REFERENCES movies(imdb_id),
            watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_name, imdb_id)
        );

        CREATE TABLE IF NOT EXISTS omdb_cache (
            imdb_id TEXT PRIMARY KEY,
            data TEXT NOT NULL  -- full JSON response
        );

        CREATE INDEX IF NOT EXISTS idx_watch_profile ON watch_history(profile_name);
    """)
    conn.commit()
    conn.close()


# --- Movies ---

def upsert_movie(movie: dict):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO movies
           (imdb_id, title, year, rated, runtime, genres, director, actors, plot, poster, imdb_rating, imdb_votes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            movie["imdb_id"],
            movie.get("title", ""),
            movie.get("year", ""),
            movie.get("rated", ""),
            movie.get("runtime", ""),
            json.dumps(movie.get("genres", [])),
            movie.get("director", ""),
            json.dumps(movie.get("actors", [])),
            movie.get("plot", ""),
            movie.get("poster", ""),
            movie.get("imdb_rating", ""),
            movie.get("imdb_votes", ""),
        ),
    )
    conn.commit()
    conn.close()


def upsert_movies_batch(movies: list[dict]):
    conn = get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO movies
           (imdb_id, title, year, rated, runtime, genres, director, actors, plot, poster, imdb_rating, imdb_votes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                m["imdb_id"],
                m.get("title", ""),
                m.get("year", ""),
                m.get("rated", ""),
                m.get("runtime", ""),
                json.dumps(m.get("genres", [])),
                m.get("director", ""),
                json.dumps(m.get("actors", [])),
                m.get("plot", ""),
                m.get("poster", ""),
                m.get("imdb_rating", ""),
                m.get("imdb_votes", ""),
            )
            for m in movies
        ],
    )
    conn.commit()
    conn.close()


def _row_to_movie(row: sqlite3.Row) -> dict:
    return {
        "imdb_id": row["imdb_id"],
        "title": row["title"],
        "year": row["year"],
        "rated": row["rated"],
        "runtime": row["runtime"],
        "genres": json.loads(row["genres"]),
        "director": row["director"],
        "actors": json.loads(row["actors"]),
        "plot": row["plot"],
        "poster": row["poster"],
        "imdb_rating": row["imdb_rating"],
        "imdb_votes": row["imdb_votes"],
    }


def load_all_movies() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM movies").fetchall()
    conn.close()
    return [_row_to_movie(r) for r in rows]


def movie_exists(imdb_id: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM movies WHERE imdb_id=?", (imdb_id,)).fetchone()
    conn.close()
    return row is not None


# --- Profiles ---

def upsert_profile(name: str, fingerprint: list[float] | None = None, fidelity: float | None = None, evolution: list | None = None):
    conn = get_conn()
    # Check if exists
    existing = conn.execute("SELECT 1 FROM profiles WHERE name=?", (name,)).fetchone()
    if existing:
        updates = []
        params = []
        if fingerprint is not None:
            updates.append("fingerprint=?")
            params.append(json.dumps(fingerprint))
        if fidelity is not None:
            updates.append("fidelity=?")
            params.append(fidelity)
        if evolution is not None:
            updates.append("evolution=?")
            params.append(json.dumps(evolution))
        if updates:
            params.append(name)
            conn.execute(f"UPDATE profiles SET {', '.join(updates)} WHERE name=?", params)
    else:
        conn.execute(
            "INSERT INTO profiles (name, fingerprint, fidelity, evolution) VALUES (?, ?, ?, ?)",
            (name, json.dumps(fingerprint) if fingerprint else None, fidelity, json.dumps(evolution or [])),
        )
    conn.commit()
    conn.close()


def delete_profile_db(name: str) -> bool:
    conn = get_conn()
    cur = conn.execute("DELETE FROM profiles WHERE name=?", (name,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def load_all_profiles() -> dict[str, dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM profiles").fetchall()
    profiles = {}
    for row in rows:
        name = row["name"]
        # Load watch history
        history_rows = conn.execute(
            "SELECT imdb_id FROM watch_history WHERE profile_name=? ORDER BY id", (name,)
        ).fetchall()
        profiles[name] = {
            "history": [r["imdb_id"] for r in history_rows],
            "fingerprint": json.loads(row["fingerprint"]) if row["fingerprint"] else None,
            "fidelity": row["fidelity"],
            "evolution": json.loads(row["evolution"]) if row["evolution"] else [],
        }
    conn.close()
    return profiles


def list_profile_names() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT name FROM profiles").fetchall()
    conn.close()
    return [r["name"] for r in rows]


# --- Watch History ---

def add_watch(profile_name: str, imdb_id: str) -> bool:
    """Returns True if newly added, False if already existed."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watch_history (profile_name, imdb_id) VALUES (?, ?)",
            (profile_name, imdb_id),
        )
        conn.commit()
        added = conn.total_changes > 0
    except Exception:
        added = False
    conn.close()
    return added


def get_watch_history(profile_name: str) -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT imdb_id FROM watch_history WHERE profile_name=? ORDER BY id", (profile_name,)
    ).fetchall()
    conn.close()
    return [r["imdb_id"] for r in rows]


# --- OMDb Cache ---

def get_omdb_cached(imdb_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT data FROM omdb_cache WHERE imdb_id=?", (imdb_id,)).fetchone()
    conn.close()
    if row:
        return json.loads(row["data"])
    return None


def set_omdb_cached(imdb_id: str, data: dict):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO omdb_cache (imdb_id, data) VALUES (?, ?)",
        (imdb_id, json.dumps(data, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def get_all_omdb_cached_ids() -> set[str]:
    conn = get_conn()
    rows = conn.execute("SELECT imdb_id FROM omdb_cache").fetchall()
    conn.close()
    return {r["imdb_id"] for r in rows}
