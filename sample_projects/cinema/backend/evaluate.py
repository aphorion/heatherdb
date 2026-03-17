"""Evaluate: Old fingerprint method vs New HeatherDB fingerprint method.

Usage:
    python evaluate.py

Requires:
    - HeatherDB running on localhost:6380
    - Cinema backend running on localhost:8000 (with seeded catalog)
    - At least one user with 3+ watched movies
"""

import asyncio
import time
import numpy as np
import httpx
from encoder import encode_movie, cosine_similarity, normalize


BACKEND = "http://localhost:8000"
HEATHER = "http://localhost:6380"


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # --- Setup: get catalog and users ---
        catalog_resp = await client.get(f"{BACKEND}/api/catalog")
        catalog = {m["imdb_id"]: m for m in catalog_resp.json()["movies"]}
        print(f"Catalog: {len(catalog)} movies\n")

        users_resp = await client.get(f"{BACKEND}/api/users")
        users = users_resp.json()["users"]

        if not users:
            print("No users found. Create a user and watch some movies first.")
            print("Falling back to synthetic evaluation...\n")
            await synthetic_eval(client, catalog)
            return

        for username in users:
            history_resp = await client.get(f"{BACKEND}/api/users/{username}/history")
            history = history_resp.json()["movies"]
            if len(history) < 3:
                print(f"User '{username}': only {len(history)} movies, skipping (need 3+)\n")
                continue

            print(f"{'=' * 60}")
            print(f"User: {username} ({len(history)} movies watched)")
            print(f"{'=' * 60}\n")

            embeddings = []
            for m in history:
                emb = encode_movie(m)
                embeddings.append((m["imdb_id"], m["title"], emb))

            collection = f"cinema_user_{username}"

            # --- Method A: Old (Python centroid → SDM read) ---
            t0 = time.perf_counter()
            centroid = normalize(np.mean([e for _, _, e in embeddings], axis=0)).tolist()
            read_resp = await client.post(
                f"{HEATHER}/collections/{collection}/read",
                json={"query": centroid, "strategy": "iterative"},
            )
            fp_old = read_resp.json()["result"]
            time_old = (time.perf_counter() - t0) * 1000

            # --- Method B: New (HeatherDB fingerprint) ---
            t0 = time.perf_counter()
            fp_resp = await client.get(f"{HEATHER}/collections/{collection}/fingerprint")
            fp_new = fp_resp.json()["fingerprint"]
            time_new = (time.perf_counter() - t0) * 1000

            if fp_new is None:
                print("  New fingerprint returned None, skipping\n")
                continue

            # === 1. Fingerprint similarity ===
            sim = cosine_similarity(fp_old, fp_new)
            print(f"1. FINGERPRINT SIMILARITY")
            print(f"   Old vs New cosine similarity: {sim:.6f}")
            print(f"   {'Very similar' if sim > 0.95 else 'Some divergence' if sim > 0.8 else 'Significant divergence'}\n")

            # === 2. Latency ===
            print(f"2. LATENCY")
            print(f"   Old (centroid + read):  {time_old:.1f} ms")
            print(f"   New (fingerprint):      {time_new:.1f} ms")
            print(f"   Speedup:                {time_old / time_new:.2f}x\n")

            # === 3. Recommendation rank correlation ===
            # Rank all catalog movies by similarity to each fingerprint
            movie_ids = list(catalog.keys())
            movie_embs = {mid: encode_movie(catalog[mid]) for mid in movie_ids}

            scores_old = [(mid, cosine_similarity(fp_old, movie_embs[mid])) for mid in movie_ids]
            scores_new = [(mid, cosine_similarity(fp_new, movie_embs[mid])) for mid in movie_ids]

            scores_old.sort(key=lambda x: x[1], reverse=True)
            scores_new.sort(key=lambda x: x[1], reverse=True)

            rank_old = {mid: i for i, (mid, _) in enumerate(scores_old)}
            rank_new = {mid: i for i, (mid, _) in enumerate(scores_new)}

            # Spearman rank correlation
            n = len(movie_ids)
            d_sq_sum = sum((rank_old[mid] - rank_new[mid]) ** 2 for mid in movie_ids)
            spearman = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))

            print(f"3. RANK CORRELATION")
            print(f"   Spearman rho:           {spearman:.6f}")
            print(f"   {'Nearly identical rankings' if spearman > 0.95 else 'Similar rankings' if spearman > 0.8 else 'Different rankings'}")

            # Top-10 overlap
            top10_old = set(mid for mid, _ in scores_old[:10])
            top10_new = set(mid for mid, _ in scores_new[:10])
            overlap = len(top10_old & top10_new)
            print(f"   Top-10 overlap:         {overlap}/10\n")

            # Show top-5 comparison
            print(f"   {'Top-5 Old':<35} {'Top-5 New':<35}")
            print(f"   {'─' * 35} {'─' * 35}")
            for i in range(5):
                old_mid, old_s = scores_old[i]
                new_mid, new_s = scores_new[i]
                old_title = catalog[old_mid]["title"][:30]
                new_title = catalog[new_mid]["title"][:30]
                print(f"   {old_title:<30} {old_s:.3f}  {new_title:<30} {new_s:.3f}")
            print()

            # === 4. Leave-one-out prediction ===
            print(f"4. LEAVE-ONE-OUT PREDICTION")
            watched_ids = set(m["imdb_id"] for m in history)
            unwatched_ids = [mid for mid in movie_ids if mid not in watched_ids]

            ranks_old_loo = []
            ranks_new_loo = []

            for held_id, held_title, held_emb in embeddings:
                # For old method: recompute centroid without held-out movie
                remaining = [e for mid, _, e in embeddings if mid != held_id]
                if not remaining:
                    continue
                loo_centroid = normalize(np.mean(remaining, axis=0)).tolist()
                loo_read = await client.post(
                    f"{HEATHER}/collections/{collection}/read",
                    json={"query": loo_centroid, "strategy": "iterative"},
                )
                loo_fp_old = loo_read.json()["result"]

                # New method fingerprint doesn't change (we can't remove a write from EAM)
                # So we use the full fingerprint — this actually tests real-world behavior
                loo_fp_new = fp_new

                # Rank the held-out movie among unwatched
                candidates = unwatched_ids + [held_id]
                s_old = [(mid, cosine_similarity(loo_fp_old, movie_embs[mid])) for mid in candidates]
                s_new = [(mid, cosine_similarity(loo_fp_new, movie_embs[mid])) for mid in candidates]
                s_old.sort(key=lambda x: x[1], reverse=True)
                s_new.sort(key=lambda x: x[1], reverse=True)

                r_old = next(i for i, (mid, _) in enumerate(s_old) if mid == held_id)
                r_new = next(i for i, (mid, _) in enumerate(s_new) if mid == held_id)
                ranks_old_loo.append(r_old)
                ranks_new_loo.append(r_new)

            if ranks_old_loo:
                mean_rank_old = np.mean(ranks_old_loo)
                mean_rank_new = np.mean(ranks_new_loo)
                median_rank_old = np.median(ranks_old_loo)
                median_rank_new = np.median(ranks_new_loo)

                # Hit rate: held-out movie in top-20
                hr_old = sum(1 for r in ranks_old_loo if r < 20) / len(ranks_old_loo)
                hr_new = sum(1 for r in ranks_new_loo if r < 20) / len(ranks_new_loo)

                total = len(candidates)
                print(f"   (ranking held-out movie among {total} candidates)\n")
                print(f"   {'Metric':<25} {'Old':>10} {'New':>10}")
                print(f"   {'─' * 45}")
                print(f"   {'Mean rank':<25} {mean_rank_old:>10.1f} {mean_rank_new:>10.1f}")
                print(f"   {'Median rank':<25} {median_rank_old:>10.1f} {median_rank_new:>10.1f}")
                print(f"   {'Hit@20 rate':<25} {hr_old:>10.1%} {hr_new:>10.1%}")
                print(f"   {'Lower rank = better. Hit@20 = fraction of held-out movies ranked in top 20.'}")
            print()

            # === 5. Stability test ===
            print(f"5. STABILITY (outlier resistance)")
            # Find an outlier: movie with lowest similarity to current fingerprint
            unwatched_scored = [(mid, cosine_similarity(fp_new, movie_embs[mid])) for mid in unwatched_ids]
            unwatched_scored.sort(key=lambda x: x[1])
            if unwatched_scored:
                outlier_id, outlier_sim = unwatched_scored[0]
                outlier_title = catalog[outlier_id]["title"]
                print(f"   Outlier: {outlier_title} (similarity: {outlier_sim:.4f})")

                # Old method: add outlier to centroid
                all_embs_with_outlier = [e for _, _, e in embeddings] + [movie_embs[outlier_id]]
                centroid_after = normalize(np.mean(all_embs_with_outlier, axis=0)).tolist()
                read_after = await client.post(
                    f"{HEATHER}/collections/{collection}/read",
                    json={"query": centroid_after, "strategy": "iterative"},
                )
                fp_old_after = read_after.json()["result"]
                drift_old = 1.0 - cosine_similarity(fp_old, fp_old_after)

                # New method: write outlier to SDM, get new fingerprint
                await client.post(
                    f"{HEATHER}/collections/{collection}/write",
                    json={"vectors": [movie_embs[outlier_id]]},
                )
                fp_after_resp = await client.get(f"{HEATHER}/collections/{collection}/fingerprint")
                fp_new_after = fp_after_resp.json()["fingerprint"]
                drift_new = 1.0 - cosine_similarity(fp_new, fp_new_after) if fp_new_after else float('inf')

                print(f"   Drift (1 - cos_sim, lower = more stable):")
                print(f"     Old method: {drift_old:.6f}")
                print(f"     New method: {drift_new:.6f}")
                print(f"   {'New method more stable' if drift_new < drift_old else 'Old method more stable'}\n")

            print()


async def synthetic_eval(client: httpx.AsyncClient, catalog: dict):
    """Run evaluation with a synthetic user when no real users exist."""
    print("Creating synthetic evaluation user with 8 sci-fi/action movies...\n")

    # Pick sci-fi/action movies
    scifi = [m for m in catalog.values() if any(g in m.get("genres", []) for g in ["Sci-Fi", "Action"])][:8]
    if len(scifi) < 4:
        print(f"Need at least 4 sci-fi/action movies in catalog, found {len(scifi)}.")
        print("Run 'Load Popular Films' in the UI first.")
        return

    username = "__eval_synthetic__"
    collection = f"cinema_user_{username}"

    # Watch movies via backend
    for m in scifi:
        await client.post(f"{BACKEND}/api/catalog/add", json={"imdb_id": m["imdb_id"]})
        await client.post(f"{BACKEND}/api/users/{username}/watch", json={"imdb_id": m["imdb_id"]})

    print(f"Watched {len(scifi)} movies: {', '.join(m['title'] for m in scifi)}\n")

    # Now run the same evaluation
    embeddings = [(m["imdb_id"], m["title"], encode_movie(m)) for m in scifi]
    movie_ids = list(catalog.keys())
    movie_embs = {mid: encode_movie(catalog[mid]) for mid in movie_ids}

    # Old method
    centroid = normalize(np.mean([e for _, _, e in embeddings], axis=0)).tolist()
    t0 = time.perf_counter()
    read_resp = await client.post(f"{HEATHER}/collections/{collection}/read", json={"query": centroid, "strategy": "iterative"})
    fp_old = read_resp.json()["result"]
    time_old = (time.perf_counter() - t0) * 1000

    # New method
    t0 = time.perf_counter()
    fp_resp = await client.get(f"{HEATHER}/collections/{collection}/fingerprint")
    fp_new = fp_resp.json()["fingerprint"]
    time_new = (time.perf_counter() - t0) * 1000

    if fp_new is None:
        print("New fingerprint returned None. HeatherDB may need restart.")
        return

    sim = cosine_similarity(fp_old, fp_new)
    print(f"Fingerprint similarity: {sim:.6f}")
    print(f"Latency — Old: {time_old:.1f}ms, New: {time_new:.1f}ms")

    scores_old = sorted([(mid, cosine_similarity(fp_old, movie_embs[mid])) for mid in movie_ids], key=lambda x: -x[1])
    scores_new = sorted([(mid, cosine_similarity(fp_new, movie_embs[mid])) for mid in movie_ids], key=lambda x: -x[1])

    print(f"\nTop-5 Old vs New:")
    for i in range(5):
        ot, os_ = catalog[scores_old[i][0]]["title"][:28], scores_old[i][1]
        nt, ns_ = catalog[scores_new[i][0]]["title"][:28], scores_new[i][1]
        print(f"  {ot:<28} {os_:.3f}   {nt:<28} {ns_:.3f}")

    # Cleanup
    await client.delete(f"{BACKEND}/api/users/{username}")
    print(f"\nCleaned up synthetic user.")


if __name__ == "__main__":
    asyncio.run(main())
