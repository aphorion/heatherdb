#!/usr/bin/env python3
"""Entry point for Navigator — robot learning through SDM."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from navigator.client import HeatherClient
from navigator.brain import Brain
from navigator import cli


def main():
    heather = HeatherClient()

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --data-dir /tmp/nav_db --dimension 128 --port 6380")
        sys.exit(1)
    print("OK")

    brain = Brain(heather)

    try:
        cli.run(brain)
    finally:
        heather.close()


if __name__ == "__main__":
    main()
