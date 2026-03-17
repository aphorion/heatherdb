#!/usr/bin/env python3
"""Entry point for Emergence — Zero-Shot Capability via Memory Composition."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from emergence.client import HeatherClient
from emergence.embeddings import Embedder
from emergence import cli


def main():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    heather = HeatherClient(url)

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- \\")
        print("    --data-dir /tmp/emergence_db \\")
        print("    --dimension 384 \\")
        print("    --port 6380")
        sys.exit(1)

    if not api_key or api_key == "your-api-key-here":
        print("ANTHROPIC_API_KEY not set. Add it to .env:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    import anthropic
    llm = anthropic.Anthropic(api_key=api_key)

    print("Loading embedding model...")
    embedder = Embedder()
    print("Ready.\n")

    try:
        cli.run(heather, embedder, llm)
    finally:
        heather.close()


if __name__ == "__main__":
    main()
