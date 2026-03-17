#!/usr/bin/env python3
"""Entry point for Genesis — artificial life in associative memory."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from genesis.client import HeatherClient
from genesis.world import World
from genesis.narrator import Narrator
from genesis import cli


def main():
    heather = HeatherClient()

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/genesis_db --dimension 384 --port 6380")
        sys.exit(1)
    print("OK")

    world = World(heather)
    narrator = Narrator()

    try:
        cli.run(world, narrator)
    finally:
        heather.close()


if __name__ == "__main__":
    main()
