#!/usr/bin/env python3
"""Entry point for Sieve — near-duplicate detection powered by HeatherDB."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from sieve.client import HeatherClient
from sieve.embeddings import EmbeddingStore
from sieve.detector import Sieve
from sieve import cli


USAGE = """
Sieve — near-duplicate detection powered by HeatherDB

Usage:
  python run.py ingest <file> [--field <name>]   Load records into EAM
  python run.py check "some text"                Check one record
  python run.py scan <file> [--field <name>]     Find duplicates within a file
  python run.py stats                            Show statistics
  python run.py interactive                      Interactive mode
"""


def main():
    heather = HeatherClient()

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/sieve_db --dimension 384 --port 6380")
        sys.exit(1)

    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "sieve.db"))
    sieve = Sieve(heather, embeddings)

    args = sys.argv[1:]
    if not args:
        print(USAGE)
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    try:
        if command == "ingest":
            cli.cmd_ingest(sieve, rest)
        elif command == "check":
            cli.cmd_check(sieve, rest)
        elif command == "scan":
            cli.cmd_scan(sieve, rest)
        elif command == "stats":
            cli.cmd_stats(sieve)
        elif command == "interactive":
            cli.cmd_interactive(sieve)
        else:
            print(f"Unknown command: {command}")
            print(USAGE)
            sys.exit(1)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
