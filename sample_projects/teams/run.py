#!/usr/bin/env python3
"""
Team Knowledge Base Demo
========================

Demonstrates N-way confidence-routed composition: each team member has
their own SDM collection. Queries route to whoever knows the answer,
with routing weights showing attribution.

Prerequisites:
    pip install -r requirements.txt
    cargo run --release -p heather_server -- --data-dir /tmp/teams_db --dimension 384
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from teams.client import HeatherClient
from teams.embeddings import Embedder
from teams.knowledge import TEAM, QUERIES, QUERIES_WITH_DAVE


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_weights(weights: dict[str, float], roles: dict[str, str]):
    """Print routing weights as a bar chart."""
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    for name, w in sorted_w:
        bar = "#" * int(w * 40)
        role = roles.get(name, "")
        print(f"  {name:>8} ({role:<20}) {bar} {w:.1%}")


def build_team(
    client: HeatherClient,
    embedder: Embedder,
    members: list[str],
):
    """Write each team member's knowledge into their own collection."""
    for name in members:
        info = TEAM[name]
        client.create_collection(name)
        vecs = embedder.embed_batch(info["snippets"])
        metadata = [{"text": s, "member": name} for s in info["snippets"]]
        result = client.write(name, vecs, metadata=metadata)
        stats = client.stats(name)
        print(
            f"  {name} ({info['role']}): "
            f"{len(info['snippets'])} snippets -> "
            f"{stats['num_locations']} locations"
        )


def query_team(
    client: HeatherClient,
    embedder: Embedder,
    members: list[str],
    queries: list[str],
    routing_sharpness: float = 20.0,
):
    """Query the composed team and display routing weights."""
    roles = {name: TEAM[name]["role"] for name in members}

    for query in queries:
        print(f"  Q: {query}")
        q_vec = embedder.embed(query)
        result = client.compose_read(members, q_vec, routing_sharpness)
        print_weights(result["weights"], roles)
        print()


def sharpness_demo(
    client: HeatherClient,
    embedder: Embedder,
    members: list[str],
):
    """Show how routing sharpness affects weight distribution."""
    query = "How do we deploy the React frontend to production?"
    q_vec = embedder.embed(query)
    roles = {name: TEAM[name]["role"] for name in members}

    print(f"  Q: {query}\n")
    for sharpness in [1.0, 5.0, 20.0, 50.0]:
        result = client.compose_read(members, q_vec, sharpness)
        sorted_w = sorted(result["weights"].items(), key=lambda x: -x[1])
        weights_str = ", ".join(f"{n}={w:.1%}" for n, w in sorted_w)
        print(f"  sharpness={sharpness:5.1f}  ->  {weights_str}")


def cleanup(client: HeatherClient, members: list[str]):
    """Drop all team collections."""
    for name in members:
        try:
            client.drop_collection(name)
        except Exception:
            pass


def main():
    url = os.environ.get("HEATHER_URL", "http://localhost:6380")
    client = HeatherClient(url)
    embedder = Embedder()

    if not client.health():
        print(f"HeatherDB not reachable at {url}")
        print("Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/teams_db --dimension 384")
        sys.exit(1)

    initial_members = ["alice", "bob", "carol"]
    all_members = ["alice", "bob", "carol", "dave"]

    # Clean slate
    cleanup(client, all_members)

    # ── Step 1: Build the team ──
    print_header("Step 1: Build the Team")
    build_team(client, embedder, initial_members)

    # ── Step 2: Query the composed team ──
    print_header("Step 2: Query the Composed Team")
    query_team(client, embedder, initial_members, QUERIES)

    # ── Step 3: Add Dave (Security) ──
    print_header("Step 3: Add Dave (Security Engineer)")
    print("  Adding Dave to the team...\n")
    build_team(client, embedder, ["dave"])
    print()
    print("  Re-querying with Dave included:\n")
    query_team(client, embedder, all_members, QUERIES_WITH_DAVE)

    # ── Step 4: Routing sharpness ──
    print_header("Step 4: Routing Sharpness Comparison")
    sharpness_demo(client, embedder, all_members)

    # Cleanup
    cleanup(client, all_members)
    client.close()
    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
