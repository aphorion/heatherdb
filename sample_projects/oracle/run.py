"""Entry point for Oracle — diagnostic pattern completion."""

import os
from dotenv import load_dotenv

load_dotenv()

from oracle.client import HeatherClient
from oracle.engine import Oracle
from oracle.cli import run


def main():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    heather = HeatherClient(url)

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- "
              "--data-dir /tmp/oracle_db --dimension 64 --port 6380")
        return

    oracle = Oracle(heather)

    try:
        run(oracle)
    finally:
        heather.close()


if __name__ == "__main__":
    main()
