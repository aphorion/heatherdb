#!/usr/bin/env python3
"""Entry point for Memoria — AI agent with associative memory."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

from memoria.client import HeatherClient
from memoria.embeddings import EmbeddingStore
from memoria.agent import Memoria
from memoria import cli


def main():
    heather = HeatherClient()

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/memoria_db --dimension 384 --port 6380")
        sys.exit(1)
    print("OK")

    print("Loading embedding model...", end=" ", flush=True)
    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "memoria.db"))
    print("OK")

    agent = Memoria(heather, embeddings)

    try:
        cli.run(agent)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
