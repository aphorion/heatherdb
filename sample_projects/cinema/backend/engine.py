"""Recommendation engine — manages catalog, users, fingerprints."""

from __future__ import annotations

import numpy as np

from heather import HeatherClient
from encoder import encode_movie, cosine_similarity, genre_affinities, normalize

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Sport", "Thriller", "War", "Western",
]


def _user_collection(username: str) -> str:
    """HeatherDB collection name for a user."""
    return f"cinema_user_{username}"


class CinemaEngine:
    def __init__(self, heather: HeatherClient):
        self.heather = heather
        # In-memory stores (fine for demo)
        self.catalog: dict[str, dict] = {}  # imdb_id → movie + embedding
        self.users: dict[str, dict] = {}    # username → {history: [...], fingerprint: [...]}

    def add_to_catalog(self, movie: dict) -> list[float]:
        """Encode movie and store in catalog."""
        embedding = encode_movie(movie)
        self.catalog[movie["imdb_id"]] = {**movie, "embedding": embedding}
        return embedding

    def get_catalog(self) -> list[dict]:
        """Return all catalog movies (without embeddings for API response)."""
        results = []
        for m in self.catalog.values():
            out = {k: v for k, v in m.items() if k != "embedding"}
            results.append(out)
        return results

    def _ensure_user(self, username: str):
        if username not in self.users:
            self.users[username] = {"history": [], "fingerprint": None, "fidelity": None, "evolution": []}

    async def watch_movie(self, username: str, imdb_id: str) -> bool:
        """User watches a movie — write embedding to HeatherDB."""
        if imdb_id not in self.catalog:
            return False
        self._ensure_user(username)
        movie = self.catalog[imdb_id]
        embedding = movie["embedding"]

        # Write to user's HeatherDB collection with metadata
        collection = _user_collection(username)
        await self.heather.write(collection, [embedding], metadata=[{
            "imdb_id": imdb_id,
            "title": movie.get("title", imdb_id),
            "genres": movie.get("genres", []),
        }])

        # Track in user history
        if imdb_id not in self.users[username]["history"]:
            self.users[username]["history"].append(imdb_id)

        # Capture previous fingerprint for evolution tracking
        prev_fp = self.users[username]["fingerprint"]

        # Update fingerprint
        await self._update_fingerprint(username)

        # Record evolution snapshot
        new_fp = self.users[username]["fingerprint"]
        if new_fp is not None:
            # Similarity to previous fingerprint (drift measurement)
            drift_sim = cosine_similarity(prev_fp, new_fp) if prev_fp is not None else 0.0

            # Analyze to get Hopfield convergence info
            try:
                trace = await self.heather.analyze(collection, new_fp)
                iterations = trace.get("iterations", 0)
                converged = trace.get("converged", False)
            except Exception:
                iterations = 0
                converged = True

            affinities = genre_affinities(new_fp, ALL_GENRES)
            top_genres = dict(list(affinities.items())[:5])

            self.users[username]["evolution"].append({
                "step": len(self.users[username]["evolution"]) + 1,
                "movie_watched": imdb_id,
                "movie_title": movie.get("title", imdb_id),
                "fingerprint_stability": round(drift_sim, 4),
                "hopfield_iterations": iterations,
                "converged": converged,
                "top_genres": top_genres,
            })

        return True

    async def _update_fingerprint(self, username: str):
        """Recompute taste fingerprint directly from HeatherDB."""
        user = self.users[username]
        if not user["history"]:
            user["fingerprint"] = None
            user["fidelity"] = None
            return

        collection = _user_collection(username)

        # Get fingerprint directly from HeatherDB
        # (weighted centroid of hard locations → Hopfield refinement)
        fingerprint = await self.heather.fingerprint(collection)
        if fingerprint is None:
            user["fingerprint"] = None
            user["fidelity"] = None
            return

        user["fingerprint"] = fingerprint

        # Compute centroid locally just for fidelity comparison
        embeddings = [self.catalog[mid]["embedding"] for mid in user["history"] if mid in self.catalog]
        if not embeddings:
            user["fidelity"] = None
            return

        centroid = normalize(np.mean(embeddings, axis=0)).tolist()
        fingerprint_fidelity = cosine_similarity(centroid, fingerprint)

        # Per-movie fidelity: read back each movie embedding from EAM
        movie_fidelities = {}
        for imdb_id in user["history"]:
            if imdb_id in self.catalog:
                emb = self.catalog[imdb_id]["embedding"]
                readback = await self.heather.read(collection, emb)
                movie_fidelities[imdb_id] = round(cosine_similarity(emb, readback), 4)

        user["fidelity"] = {
            "fingerprint": round(fingerprint_fidelity, 4),
            "per_movie": movie_fidelities,
            "mean_movie": round(
                float(np.mean(list(movie_fidelities.values()))) if movie_fidelities else 0.0, 4
            ),
        }

    def get_history(self, username: str) -> list[dict]:
        """Return user's watched movies."""
        self._ensure_user(username)
        result = []
        for imdb_id in self.users[username]["history"]:
            if imdb_id in self.catalog:
                m = self.catalog[imdb_id]
                result.append({k: v for k, v in m.items() if k != "embedding"})
        return result

    def get_fingerprint(self, username: str) -> dict | None:
        """Return user's taste fingerprint + genre affinities + fidelity."""
        self._ensure_user(username)
        fp = self.users[username]["fingerprint"]
        if fp is None:
            return None
        affinities = genre_affinities(fp, ALL_GENRES)
        return {
            "vector": fp,
            "genre_affinities": affinities,
            "movies_watched": len(self.users[username]["history"]),
            "fidelity": self.users[username].get("fidelity"),
        }

    def recommend(self, username: str, n: int = 10, threshold: float = 0.0) -> tuple[list[dict], dict | None]:
        """Rank catalog movies by similarity to user fingerprint. Returns (recommendations, fidelity)."""
        self._ensure_user(username)
        fp = self.users[username]["fingerprint"]
        if fp is None:
            return [], None

        watched = set(self.users[username]["history"])
        fidelity_data = self.users[username].get("fidelity")
        movie_fidelities = fidelity_data["per_movie"] if fidelity_data else {}

        scored = []
        for imdb_id, movie in self.catalog.items():
            sim = cosine_similarity(fp, movie["embedding"])
            is_watched = imdb_id in watched
            if sim >= threshold:
                out = {k: v for k, v in movie.items() if k != "embedding"}
                out["similarity"] = round(sim, 4)
                out["watched"] = is_watched
                if is_watched and imdb_id in movie_fidelities:
                    out["fidelity"] = movie_fidelities[imdb_id]
                scored.append(out)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:n], fidelity_data

    async def explain_recommendation(self, username: str, imdb_id: str) -> dict:
        """Explain why a movie was recommended by tracing SDM activations."""
        self._ensure_user(username)
        user = self.users[username]
        if user["fingerprint"] is None:
            return {"error": "User has no fingerprint yet"}
        if imdb_id not in self.catalog:
            return {"error": "Movie not in catalog"}

        collection = _user_collection(username)
        movie = self.catalog[imdb_id]

        # Analyze: query the recommended movie's embedding against user's collection
        trace = await self.heather.analyze(collection, movie["embedding"])
        activated = trace.get("activated_locations", [])
        activated_ids = {loc["id"]: loc["weight"] for loc in activated}

        # Batch analyze all watched movies in a single call
        watched_ids = [wid for wid in user["history"] if wid in self.catalog]
        watched_embeddings = [self.catalog[wid]["embedding"] for wid in watched_ids]

        movie_contributions: dict[str, float] = {}
        if watched_embeddings:
            watched_traces = await self.heather.batch_analyze(collection, watched_embeddings)
            for watched_id, watched_trace in zip(watched_ids, watched_traces):
                watched_locs = watched_trace.get("activated_locations", [])

                # Contribution = sum of shared location weights
                overlap_weight = 0.0
                for loc in watched_locs:
                    if loc["id"] in activated_ids:
                        overlap_weight += activated_ids[loc["id"]] * loc["weight"]
                if overlap_weight > 0:
                    movie_contributions[watched_id] = overlap_weight

        # Normalize to percentages
        total = sum(movie_contributions.values())
        explanations = []
        if total > 0:
            for mid, weight in sorted(movie_contributions.items(), key=lambda x: -x[1]):
                m = self.catalog[mid]
                pct = weight / total
                explanations.append({
                    "imdb_id": mid,
                    "title": m.get("title", mid),
                    "contribution": round(pct, 4),
                    "percentage": f"{pct * 100:.0f}%",
                })

        rec_movie = {k: v for k, v in movie.items() if k != "embedding"}
        return {
            "movie": rec_movie,
            "explanations": explanations,
            "hopfield_iterations": trace.get("iterations", 0),
            "converged": trace.get("converged", False),
            "activated_location_count": len(activated),
        }

    def get_taste_evolution(self, username: str) -> dict:
        """Return the user's taste evolution timeline."""
        self._ensure_user(username)
        evolution = self.users[username]["evolution"]
        return {
            "username": username,
            "total_watches": len(evolution),
            "timeline": evolution,
        }

    def blend_fingerprints(self, user1: str, user2: str, n: int = 10) -> dict:
        """Blend two users' fingerprints and recommend."""
        self._ensure_user(user1)
        self._ensure_user(user2)

        fp1 = self.users[user1].get("fingerprint")
        fp2 = self.users[user2].get("fingerprint")

        if fp1 is None or fp2 is None:
            return {"error": "Both users need fingerprints", "recommendations": []}

        # Blend = average of fingerprints
        blended = normalize(
            np.array(fp1, dtype=np.float64) + np.array(fp2, dtype=np.float64)
        ).tolist()

        similarity = cosine_similarity(fp1, fp2)

        # Rank catalog against blended
        watched1 = set(self.users[user1]["history"])
        watched2 = set(self.users[user2]["history"])

        scored = []
        for imdb_id, movie in self.catalog.items():
            sim = cosine_similarity(blended, movie["embedding"])
            out = {k: v for k, v in movie.items() if k != "embedding"}
            out["similarity"] = round(sim, 4)
            out["watched_by"] = []
            if imdb_id in watched1:
                out["watched_by"].append(user1)
            if imdb_id in watched2:
                out["watched_by"].append(user2)
            scored.append(out)

        scored.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "user1_affinities": genre_affinities(fp1, ALL_GENRES),
            "user2_affinities": genre_affinities(fp2, ALL_GENRES),
            "taste_similarity": round(similarity, 4),
            "recommendations": scored[:n],
        }

    def delete_user(self, username: str) -> bool:
        """Remove user data."""
        if username in self.users:
            del self.users[username]
            return True
        return False

    def list_users(self) -> list[str]:
        return list(self.users.keys())

    async def demo_stability(self, outlier_id: str | None = None) -> dict:
        """Demo: show fingerprint stability when adding one outlier."""
        test_user = "__stability_demo__"
        self._ensure_user(test_user)

        # Collect sci-fi movies from catalog
        scifi = [
            m for m in self.catalog.values()
            if "Sci-Fi" in m.get("genres", []) or "Action" in m.get("genres", [])
        ][:5]

        if len(scifi) < 2:
            return {"error": "Need at least 2 sci-fi/action movies in catalog"}

        # Write sci-fi movies
        self.users[test_user] = {"history": [], "fingerprint": None}
        for m in scifi:
            await self.watch_movie(test_user, m["imdb_id"])

        before_fp = self.users[test_user]["fingerprint"]
        before_recs, _ = self.recommend(test_user, n=5)

        # Add outlier if provided
        if outlier_id and outlier_id in self.catalog:
            await self.watch_movie(test_user, outlier_id)

        after_fp = self.users[test_user]["fingerprint"]
        after_recs, _ = self.recommend(test_user, n=5)

        sim = cosine_similarity(before_fp, after_fp) if before_fp and after_fp else 1.0

        # Cleanup
        del self.users[test_user]

        return {
            "fingerprint_similarity": round(sim, 4),
            "before_recommendations": before_recs,
            "after_recommendations": after_recs,
            "sci_fi_count": len(scifi),
            "outlier_added": outlier_id is not None,
        }

    async def demo_coldstart(self) -> dict:
        """Demo: show how recommendations improve with more watches."""
        test_user = "__coldstart_demo__"
        self.users[test_user] = {"history": [], "fingerprint": None}

        movies = list(self.catalog.values())[:8]
        if len(movies) < 3:
            return {"error": "Need at least 3 movies in catalog"}

        snapshots = []
        for i, m in enumerate(movies):
            await self.watch_movie(test_user, m["imdb_id"])
            recs, _ = self.recommend(test_user, n=5)
            fp_data = self.get_fingerprint(test_user)
            snapshots.append({
                "movies_watched": i + 1,
                "recommendations": recs,
                "top_genres": dict(list(fp_data["genre_affinities"].items())[:5]) if fp_data else {},
            })

        del self.users[test_user]
        return {"snapshots": snapshots}
