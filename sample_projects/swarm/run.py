#!/usr/bin/env python3
"""Entry point for Swarm — emergent intelligence from shared memory."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from swarm.client import HeatherClient
from swarm.embeddings import EmbeddingStore
from swarm.swarm import Swarm
from swarm import cli


def main():
    heather = HeatherClient()

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/swarm_db --dimension 384 --port 6380")
        sys.exit(1)
    print("OK")

    print("Loading embedding model...", end=" ", flush=True)
    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "swarm.db"))
    print("OK")

    swarm = Swarm(heather, embeddings)

    try:
        cli.run(swarm)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
