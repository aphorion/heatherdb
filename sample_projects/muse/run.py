#!/usr/bin/env python3
"""Entry point for Muse — creative idea blender powered by HeatherDB."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from muse.client import HeatherClient
from muse.embeddings import EmbeddingStore
from muse.blender import Muse
from muse import cli

SEED_CONCEPTS = [
    "jazz improvisation and syncopated rhythms",
    "gothic cathedral architecture with flying buttresses",
    "bioluminescent deep sea creatures",
    "fractal geometry in nature",
    "ancient Japanese tea ceremony rituals",
    "quantum entanglement and superposition",
    "street art and graffiti culture",
    "mycorrhizal networks connecting forest trees",
    "origami and mathematical paper folding",
    "the aurora borealis and solar winds",
]


def main():
    heather = HeatherClient()

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/muse_db --dimension 384 --port 6380")
        sys.exit(1)
    print("OK")

    print("Loading embedding model...", end=" ", flush=True)
    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "muse.db"))
    print("OK")

    muse = Muse(heather, embeddings)

    # Pre-seed concepts on first run
    if embeddings.count() == 0:
        print("Seeding starter concepts...", end=" ", flush=True)
        for concept in SEED_CONCEPTS:
            muse.add_concept(concept)
        print(f"OK ({len(SEED_CONCEPTS)} concepts)")

    try:
        cli.run(muse)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
