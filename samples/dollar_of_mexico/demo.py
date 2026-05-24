#!/usr/bin/env python3
"""
The dollar of Mexico, on a server.

Kanerva 2010, "What we mean when we say 'What's the dollar of Mexico?'".
This demo encodes four (country, currency) pairs as HRR bindings in
random 1024-d hypervectors, bundles them into one composite memory,
then asks "What is X's currency?" by unbinding the composite by the
country vector and cleaning up the noisy result against a lexicon.

Every bind, bundle, and unbind happens on the HeatherDB server, over
HTTP. The script is a thin client.

Run:
    HEATHER_BASE_URL=http://localhost:6380 \\
    HEATHER_USER=admin HEATHER_PASS=change-me-1234 \\
    python demo.py

Or, if your server runs with auth disabled (HEATHER_AUTH_DISABLED=true):
    python demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import requests

BASE = os.environ.get("HEATHER_BASE_URL", "http://localhost:6380")
USER = os.environ.get("HEATHER_USER")
PASS = os.environ.get("HEATHER_PASS")
AUTH = (USER, PASS) if USER and PASS else None
DIM = 1024
DB = "kanerva"
SEED = 7

PAIRS = [
    ("USA", "dollar"),
    ("Mexico", "peso"),
    ("Japan", "yen"),
    ("France", "euro"),
]


def rand_unit(rng: np.random.Generator) -> list[float]:
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def post(path: str, json=None):
    r = requests.post(f"{BASE}{path}", json=json, auth=AUTH)
    if not r.ok:
        sys.exit(f"POST {path} -> {r.status_code}: {r.text}")
    return r.json() if r.text else {}


def get(path: str):
    r = requests.get(f"{BASE}{path}", auth=AUTH)
    if not r.ok:
        sys.exit(f"GET {path} -> {r.status_code}: {r.text}")
    return r.json()


def delete(path: str):
    requests.delete(f"{BASE}{path}", auth=AUTH)  # 404 is fine


def bulk_load_one(name: str, vec: list[float]):
    """Make `name` a single-location collection holding exactly `vec`.
    Bypasses competitive learning so bind sees what we wrote."""
    post(f"/db/{DB}/collections", {"name": name})
    post(
        f"/db/{DB}/collections/{name}/bulk_load",
        {"addresses": [vec], "counters": [vec], "write_counts": [1.0]},
    )


def main() -> int:
    print(f"connecting to {BASE}  (auth={'on' if AUTH else 'off'})")

    # --- Setup: DB ---
    delete(f"/db/{DB}")  # idempotent reset
    post("/db", {"name": DB, "dimension": DIM})
    print(f"  db {DB!r} created (d={DIM})")

    # --- Vectors ---
    rng = np.random.default_rng(SEED)
    entities: dict[str, list[float]] = {}
    for country, currency in PAIRS:
        entities[country] = rand_unit(rng)
        entities[currency] = rand_unit(rng)
    print(f"  generated {len(entities)} random unit vectors")

    # --- Single-location entity collections ---
    for name, vec in entities.items():
        bulk_load_one(name, vec)
    print(f"  {len(entities)} entity collections bulk-loaded")

    # --- Lexicon: one collection containing all entities ---
    post(f"/db/{DB}/collections", {"name": "lexicon"})
    addresses = list(entities.values())
    post(
        f"/db/{DB}/collections/lexicon/bulk_load",
        {"addresses": addresses, "counters": addresses,
         "write_counts": [1.0] * len(addresses)},
    )
    print(f"  lexicon: {len(entities)} entries")

    # --- Bind each pair: pair_i = bind(country, currency) ---
    for i, (country, currency) in enumerate(PAIRS):
        post(
            f"/db/{DB}/algebra/bind",
            {"source_a": country, "source_b": currency, "target": f"pair_{i}"},
        )
    print(f"  {len(PAIRS)} bind operations done")

    # --- Bundle: composite = pair_0 + pair_1 + ... ---
    post(
        f"/db/{DB}/algebra/add",
        {"source_a": "pair_0", "source_b": "pair_1", "target": "composite"},
    )
    for i in range(2, len(PAIRS)):
        post(
            f"/db/{DB}/algebra/add",
            {"source_a": "composite", "source_b": f"pair_{i}",
             "target": "composite"},
        )
    print(f"  composite memory built (bundle of {len(PAIRS)} bindings)")

    # --- Query each country ---
    print()
    print("=" * 64)
    print("Q: 'What is X's currency?'")
    print("=" * 64)

    items = list(entities.items())
    item_names = [n for n, _ in items]
    item_vecs = np.array([v for _, v in items])

    failures = 0
    for country, true_currency in PAIRS:
        # Server-side unbind
        post(
            f"/db/{DB}/algebra/unbind",
            {"source": "composite", "key_vector": entities[country],
             "target": "recovered"},
        )

        # Pull the noisy recovered vector back
        locs = get(f"/db/{DB}/collections/recovered/locations?full=true")
        recovered = np.array(locs["locations"][0]["counter"])
        recovered /= np.linalg.norm(recovered)

        # Cleanup: cosine vs every entity
        sims = item_vecs @ recovered
        order = np.argsort(-sims)
        winner = item_names[order[0]]
        runner_up = item_names[order[1]]

        mark = "✓" if winner == true_currency else "✗"
        if winner != true_currency:
            failures += 1
        print(
            f"\n  {country:>7s} -> {winner:<8s} "
            f"(sim={sims[order[0]]:.3f}, next={runner_up}/{sims[order[1]]:.3f})  {mark}"
        )

    # --- Show server-side cleanup once, for the Mexico case ---
    print()
    print("=" * 64)
    print("Server-side cleanup: POST /analyze on the lexicon collection")
    print("=" * 64)
    post(
        f"/db/{DB}/algebra/unbind",
        {"source": "composite", "key_vector": entities["Mexico"],
         "target": "recovered_mx"},
    )
    locs = get(f"/db/{DB}/collections/recovered_mx/locations?full=true")
    recovered = locs["locations"][0]["counter"]
    analyze = post(
        f"/db/{DB}/collections/lexicon/analyze",
        {"query": recovered},
    )
    cleaned = np.array(analyze["result"])
    cleaned /= np.linalg.norm(cleaned)
    sims = item_vecs @ cleaned
    order = np.argsort(-sims)
    print(
        f"  Hopfield read on lexicon: {analyze['iterations']} iters, "
        f"converged={analyze['converged']}"
    )
    print(
        f"  Cleaned vector closest to: {item_names[order[0]]} "
        f"(sim={sims[order[0]]:.3f})"
    )

    print()
    if failures == 0:
        print("All four analogies recovered correctly.")
        return 0
    print(f"{failures}/{len(PAIRS)} analogies failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
