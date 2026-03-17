"""Recommendation engine — manages catalog, profiles, fingerprints.

Consumer-facing: no SDM jargon exposed. Internal terminology mapped to friendly labels.
Persistence via SQLite (db.py). Embeddings computed in-memory on load.
"""

from __future__ import annotations

import random

import numpy as np

from heather import HeatherClient
from encoder import encode_movie, cosine_similarity, genre_affinities, normalize, feature_vector, DIMENSION
import db

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Sport", "Thriller", "War", "Western",
]


def _profile_collection(name: str) -> str:
    return f"cinema2_{name}"


def _stability_label(sim: float) -> str:
    if sim >= 0.95:
        return "Stable"
    elif sim >= 0.85:
        return "Evolving"
    return "Shifting"


class CinemaEngine:
    def __init__(self, heather: HeatherClient):
        self.heather = heather
        self.catalog: dict[str, dict] = {}
        self.profiles: dict[str, dict] = {}
        db.init_db()
        self._load()

    def _load(self):
        """Load catalog and profiles from SQLite, re-encode embeddings."""
        movies = db.load_all_movies()
        for movie_data in movies:
            embedding = encode_movie(movie_data)
            self.catalog[movie_data["imdb_id"]] = {**movie_data, "embedding": embedding}
        if movies:
            print(f"[cinema2] Loaded {len(self.catalog)} movies from db")

        self.profiles = db.load_all_profiles()
        if self.profiles:
            print(f"[cinema2] Loaded {len(self.profiles)} profiles from db")

    def add_to_catalog(self, movie: dict) -> list[float]:
        embedding = encode_movie(movie)
        self.catalog[movie["imdb_id"]] = {**movie, "embedding": embedding}
        db.upsert_movie(movie)
        return embedding

    def add_to_catalog_batch(self, movies: list[dict]):
        for movie in movies:
            embedding = encode_movie(movie)
            self.catalog[movie["imdb_id"]] = {**movie, "embedding": embedding}
        db.upsert_movies_batch(movies)

    def get_catalog(self) -> list[dict]:
        return [{k: v for k, v in m.items() if k != "embedding"} for m in self.catalog.values()]

    def _ensure_profile(self, name: str):
        if name not in self.profiles:
            self.profiles[name] = {
                "history": [],
                "fingerprint": None,
                "fidelity": None,
                "evolution": [],
            }
            db.upsert_profile(name)

    # --- Watch ---

    async def watch_movie(self, name: str, imdb_id: str) -> bool:
        if imdb_id not in self.catalog:
            return False
        self._ensure_profile(name)
        movie = self.catalog[imdb_id]
        embedding = movie["embedding"]

        collection = _profile_collection(name)
        await self.heather.write(collection, [embedding], metadata=[{
            "imdb_id": imdb_id,
            "title": movie.get("title", imdb_id),
            "genres": movie.get("genres", []),
        }])

        if imdb_id not in self.profiles[name]["history"]:
            self.profiles[name]["history"].append(imdb_id)
            db.add_watch(name, imdb_id)

        prev_fp = self.profiles[name]["fingerprint"]
        await self._update_fingerprint(name)
        new_fp = self.profiles[name]["fingerprint"]

        if new_fp is not None:
            drift_sim = cosine_similarity(prev_fp, new_fp) if prev_fp is not None else 0.0
            affinities = genre_affinities(new_fp, ALL_GENRES)
            top_genres = dict(list(affinities.items())[:5])
            self.profiles[name]["evolution"].append({
                "step": len(self.profiles[name]["evolution"]) + 1,
                "movie_watched": imdb_id,
                "movie_title": movie.get("title", imdb_id),
                "stability": round(drift_sim, 4),
                "stability_label": _stability_label(drift_sim),
                "top_genres": top_genres,
            })

        # Persist profile state
        p = self.profiles[name]
        db.upsert_profile(name, fingerprint=p["fingerprint"], fidelity=p["fidelity"], evolution=p["evolution"])
        return True

    async def _update_fingerprint(self, name: str):
        profile = self.profiles[name]
        if not profile["history"]:
            profile["fingerprint"] = None
            profile["fidelity"] = None
            return

        collection = _profile_collection(name)
        fingerprint = await self.heather.fingerprint(collection)
        if fingerprint is None:
            profile["fingerprint"] = None
            profile["fidelity"] = None
            return

        profile["fingerprint"] = fingerprint

        embeddings = [self.catalog[mid]["embedding"] for mid in profile["history"] if mid in self.catalog]
        if not embeddings:
            profile["fidelity"] = None
            return

        centroid = normalize(np.mean(embeddings, axis=0)).tolist()
        profile["fidelity"] = round(cosine_similarity(centroid, fingerprint), 4)

    # --- History ---

    def get_history(self, name: str) -> list[dict]:
        self._ensure_profile(name)
        result = []
        for imdb_id in self.profiles[name]["history"]:
            if imdb_id in self.catalog:
                m = self.catalog[imdb_id]
                result.append({k: v for k, v in m.items() if k != "embedding"})
        return result

    # --- Profiles ---

    def list_profiles(self) -> list[str]:
        return list(self.profiles.keys())

    def delete_profile(self, name: str) -> bool:
        if name in self.profiles:
            del self.profiles[name]
            db.delete_profile_db(name)
            return True
        return False

    # --- Recommendations ---

    def recommend(self, name: str, n: int = 40, unwatched_only: bool = True) -> list[dict]:
        self._ensure_profile(name)
        fp = self.profiles[name]["fingerprint"]
        if fp is None:
            return []

        watched = set(self.profiles[name]["history"])
        scored = []
        for imdb_id, movie in self.catalog.items():
            if unwatched_only and imdb_id in watched:
                continue
            sim = cosine_similarity(fp, movie["embedding"])
            out = {k: v for k, v in movie.items() if k != "embedding"}
            out["match"] = round(sim * 100, 1)
            scored.append(out)

        scored.sort(key=lambda x: x["match"], reverse=True)
        return scored[:n]

    def tiered_recommendations(self, name: str, n: int = 40) -> dict:
        recs = self.recommend(name, n=n)
        tiers = {
            "perfect_for_you": [],
            "great_match": [],
            "worth_watching": [],
            "discover": [],
        }
        for r in recs:
            m = r["match"]
            if m >= 50:
                tiers["perfect_for_you"].append(r)
            elif m >= 40:
                tiers["great_match"].append(r)
            elif m >= 30:
                tiers["worth_watching"].append(r)
            else:
                tiers["discover"].append(r)
        return tiers

    # --- Because You Watched (THE KEY FEATURE) ---

    async def because_you_watched(self, name: str, max_rows: int = 5, n_per_row: int = 8) -> list[dict]:
        self._ensure_profile(name)
        fp = self.profiles[name]["fingerprint"]
        if fp is None:
            return []

        collection = _profile_collection(name)
        watched_ids = [wid for wid in self.profiles[name]["history"] if wid in self.catalog]
        if not watched_ids:
            return []

        recs = self.recommend(name, n=40)
        if not recs:
            return []

        watched_embeddings = [self.catalog[wid]["embedding"] for wid in watched_ids]
        try:
            watched_traces = await self.heather.batch_analyze(collection, watched_embeddings)
        except Exception:
            return []

        watched_weights: dict[str, dict[int, float]] = {}
        for wid, trace in zip(watched_ids, watched_traces):
            locs = trace.get("activated_locations", [])
            watched_weights[wid] = {loc["id"]: loc["weight"] for loc in locs}

        rec_embeddings = [self.catalog[r["imdb_id"]]["embedding"] for r in recs]
        try:
            rec_traces = await self.heather.batch_analyze(collection, rec_embeddings)
        except Exception:
            return []

        rows: dict[str, list[dict]] = {}
        for rec, trace in zip(recs, rec_traces):
            rec_locs = {loc["id"]: loc["weight"] for loc in trace.get("activated_locations", [])}
            best_wid = None
            best_overlap = 0.0
            for wid, w_locs in watched_weights.items():
                overlap = sum(rec_locs.get(lid, 0) * w for lid, w in w_locs.items())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_wid = wid
            if best_wid:
                rows.setdefault(best_wid, []).append(rec)

        sorted_rows = sorted(rows.items(), key=lambda x: len(x[1]), reverse=True)
        result = []
        for wid, movies in sorted_rows[:max_rows]:
            watched_movie = self.catalog[wid]
            result.append({
                "because": {
                    "imdb_id": wid,
                    "title": watched_movie.get("title", wid),
                    "poster": watched_movie.get("poster", ""),
                },
                "movies": movies[:n_per_row],
            })
        return result

    # --- Similar Movies ---

    def similar_movies(self, imdb_id: str, n: int = 10) -> list[dict]:
        if imdb_id not in self.catalog:
            return []
        target = self.catalog[imdb_id]
        scored = []
        for mid, movie in self.catalog.items():
            if mid == imdb_id:
                continue
            sim = cosine_similarity(target["embedding"], movie["embedding"])
            out = {k: v for k, v in movie.items() if k != "embedding"}
            out["match"] = round(sim * 100, 1)
            scored.append(out)
        scored.sort(key=lambda x: x["match"], reverse=True)
        return scored[:n]

    # --- Explain ---

    async def explain_recommendation(self, name: str, imdb_id: str) -> dict:
        self._ensure_profile(name)
        profile = self.profiles[name]
        if profile["fingerprint"] is None:
            return {"error": "No taste profile yet — watch some movies first!"}
        if imdb_id not in self.catalog:
            return {"error": "Movie not found"}

        collection = _profile_collection(name)
        movie = self.catalog[imdb_id]

        trace = await self.heather.analyze(collection, movie["embedding"])
        activated = trace.get("activated_locations", [])
        activated_ids = {loc["id"]: loc["weight"] for loc in activated}

        watched_ids = [wid for wid in profile["history"] if wid in self.catalog]
        watched_embeddings = [self.catalog[wid]["embedding"] for wid in watched_ids]

        contributions: dict[str, float] = {}
        if watched_embeddings:
            watched_traces = await self.heather.batch_analyze(collection, watched_embeddings)
            for wid, w_trace in zip(watched_ids, watched_traces):
                w_locs = w_trace.get("activated_locations", [])
                overlap = sum(activated_ids.get(loc["id"], 0) * loc["weight"] for loc in w_locs)
                if overlap > 0:
                    contributions[wid] = overlap

        total = sum(contributions.values())
        influences = []
        if total > 0:
            for mid, weight in sorted(contributions.items(), key=lambda x: -x[1]):
                m = self.catalog[mid]
                pct = weight / total
                influences.append({
                    "imdb_id": mid,
                    "title": m.get("title", mid),
                    "poster": m.get("poster", ""),
                    "contribution": round(pct * 100, 1),
                })

        rec_movie = {k: v for k, v in movie.items() if k != "embedding"}
        match_score = cosine_similarity(profile["fingerprint"], movie["embedding"])
        return {
            "movie": rec_movie,
            "match": round(match_score * 100, 1),
            "influences": influences,
        }

    # --- Taste Profile ---

    def get_taste_profile(self, name: str) -> dict:
        self._ensure_profile(name)
        fp = self.profiles[name]["fingerprint"]
        if fp is None:
            return {
                "status": "new",
                "genre_dna": {},
                "confidence": 0,
                "journey": [],
                "movies_watched": 0,
            }

        affinities = genre_affinities(fp, ALL_GENRES)
        genre_dna = {g: round(v * 100, 1) for g, v in affinities.items() if v > 0}

        fidelity = self.profiles[name].get("fidelity") or 0
        confidence = round(fidelity * 100, 1)

        evolution = self.profiles[name]["evolution"]
        journey = [{
            "step": e["step"],
            "movie": e["movie_title"],
            "stability": e["stability_label"],
            "top_genres": {g: round(v * 100, 1) for g, v in e["top_genres"].items()},
        } for e in evolution]

        n_watched = len(self.profiles[name]["history"])
        if n_watched < 3:
            status = "Getting to know you..."
        elif n_watched < 8:
            status = "Building your taste profile"
        else:
            status = "Your taste is well established"

        return {
            "status": status,
            "genre_dna": genre_dna,
            "confidence": confidence,
            "journey": journey,
            "movies_watched": n_watched,
        }

    # --- Genre Browsing ---

    def movies_by_genre(self, genre: str, n: int = 20, profile: str | None = None) -> list[dict]:
        fp = None
        watched = set()
        if profile and profile in self.profiles:
            fp = self.profiles[profile].get("fingerprint")
            watched = set(self.profiles[profile]["history"])

        results = []
        for imdb_id, movie in self.catalog.items():
            if genre not in movie.get("genres", []):
                continue
            if imdb_id in watched:
                continue
            out = {k: v for k, v in movie.items() if k != "embedding"}
            if fp:
                out["match"] = round(cosine_similarity(fp, movie["embedding"]) * 100, 1)
            else:
                out["match"] = round(float(movie.get("imdb_rating", "0") or "0") * 10, 1)
            results.append(out)

        results.sort(key=lambda x: x["match"], reverse=True)
        return results[:n]

    def available_genres(self) -> list[str]:
        counts: dict[str, int] = {}
        for movie in self.catalog.values():
            for g in movie.get("genres", []):
                counts[g] = counts.get(g, 0) + 1
        return sorted([g for g, c in counts.items() if c >= 3], key=lambda g: -counts[g])

    # --- Wildcards / Try Something New ---

    def wildcards(self, name: str, n: int = 10) -> list[dict]:
        self._ensure_profile(name)
        fp = self.profiles[name]["fingerprint"]
        if fp is None:
            all_movies = list(self.catalog.values())
            random.shuffle(all_movies)
            return [{k: v for k, v in m.items() if k != "embedding"} for m in all_movies[:n]]

        watched = set(self.profiles[name]["history"])
        scored = []
        for imdb_id, movie in self.catalog.items():
            if imdb_id in watched:
                continue
            sim = cosine_similarity(fp, movie["embedding"])
            out = {k: v for k, v in movie.items() if k != "embedding"}
            out["match"] = round(sim * 100, 1)
            rating = float(movie.get("imdb_rating", "0") or "0")
            out["_score"] = rating - sim
            scored.append(out)

        scored.sort(key=lambda x: x["_score"], reverse=True)
        for s in scored:
            del s["_score"]
        return scored[:n]

    # --- Continue Watching ---

    def continue_watching(self, name: str, n: int = 10) -> list[dict]:
        self._ensure_profile(name)
        history = self.profiles[name]["history"]
        result = []
        for imdb_id in reversed(history):
            if imdb_id in self.catalog:
                m = self.catalog[imdb_id]
                out = {k: v for k, v in m.items() if k != "embedding"}
                out["watched"] = True
                result.append(out)
            if len(result) >= n:
                break
        return result

    # --- Blend ---

    def blend_fingerprints(self, name1: str, name2: str, n: int = 10) -> dict:
        self._ensure_profile(name1)
        self._ensure_profile(name2)

        fp1 = self.profiles[name1].get("fingerprint")
        fp2 = self.profiles[name2].get("fingerprint")

        if fp1 is None or fp2 is None:
            return {"error": "Both profiles need watch history", "compatibility": 0, "recommendations": []}

        blended = normalize(
            np.array(fp1, dtype=np.float64) + np.array(fp2, dtype=np.float64)
        ).tolist()

        compatibility = round(cosine_similarity(fp1, fp2) * 100, 1)

        watched1 = set(self.profiles[name1]["history"])
        watched2 = set(self.profiles[name2]["history"])

        scored = []
        for imdb_id, movie in self.catalog.items():
            sim = cosine_similarity(blended, movie["embedding"])
            out = {k: v for k, v in movie.items() if k != "embedding"}
            out["match"] = round(sim * 100, 1)
            out["watched_by"] = []
            if imdb_id in watched1:
                out["watched_by"].append(name1)
            if imdb_id in watched2:
                out["watched_by"].append(name2)
            scored.append(out)

        scored.sort(key=lambda x: x["match"], reverse=True)

        return {
            "profile1_dna": {g: round(v * 100, 1) for g, v in genre_affinities(fp1, ALL_GENRES).items() if v > 0},
            "profile2_dna": {g: round(v * 100, 1) for g, v in genre_affinities(fp2, ALL_GENRES).items() if v > 0},
            "compatibility": compatibility,
            "recommendations": scored[:n],
        }
