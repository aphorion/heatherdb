#!/usr/bin/env python3
"""Entry point for Whisper — lossy semantic compression."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from whisper.client import HeatherClient
from whisper.embeddings import EmbeddingStore
from whisper.memory import Whisper
from whisper import cli


USAGE = """
Whisper — lossy semantic compression powered by HeatherDB

Usage:
  python run.py load <file>           Absorb a document into memory
  python run.py ask "question"        Query the compressed memory
  python run.py interactive           Interactive mode
  python run.py stats                 Show statistics
"""


def main():
    heather = HeatherClient()

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/whisper_db --dimension 384 --port 6380")
        sys.exit(1)

    embeddings = EmbeddingStore(db_path=str(Path(__file__).parent / "whisper.db"))
    whisp = Whisper(heather, embeddings)

    args = sys.argv[1:]
    if not args:
        print(USAGE)
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    try:
        if command == "load":
            cli.cmd_load(whisp, rest)
        elif command == "ask":
            cli.cmd_ask(whisp, rest)
        elif command == "interactive":
            cli.cmd_interactive(whisp)
        elif command == "stats":
            cli.cmd_stats(whisp)
        else:
            print(f"Unknown command: {command}")
            print(USAGE)
            sys.exit(1)
    finally:
        heather.close()
        embeddings.close()


if __name__ == "__main__":
    main()
