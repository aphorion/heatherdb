#!/usr/bin/env python3
"""Entry point for Sentinel — anomaly detection powered by HeatherDB."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from sentinel.client import HeatherClient
from sentinel.embeddings import EmbeddingStore
from sentinel.detector import Sentinel
from sentinel import cli


USAGE = """
Sentinel — anomaly detection with reconstruction diff

Usage:
  python run.py train <file>            Train on normal log patterns
  python run.py check "log line"        Check one log line
  python run.py monitor <file>          Analyze a file of log lines
  python run.py stats                   Show statistics
  python run.py interactive             Interactive mode
"""


def main():
    heather = HeatherClient()

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/sentinel_db --dimension 384 --port 6380")
        sys.exit(1)

    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "sentinel.db"))
    sentinel = Sentinel(heather, embeddings)

    args = sys.argv[1:]
    if not args:
        print(USAGE)
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    try:
        if command == "train":
            cli.cmd_train(sentinel, rest)
        elif command == "check":
            cli.cmd_check(sentinel, rest)
        elif command == "monitor":
            cli.cmd_monitor(sentinel, rest)
        elif command == "stats":
            cli.cmd_stats(sentinel)
        elif command == "interactive":
            cli.cmd_interactive(sentinel)
        else:
            print(f"Unknown command: {command}")
            print(USAGE)
            sys.exit(1)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
