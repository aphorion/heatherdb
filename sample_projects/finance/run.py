"""Entry point for Finance — Market Déjà Vu + Cross-Asset Completion."""

import os
from dotenv import load_dotenv

load_dotenv()

from finance.client import HeatherClient
from finance.engine import Finance
from finance.cli import run


def main():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    heather = HeatherClient(url)

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- "
              "--data-dir /tmp/finance_db --dimension 128 --port 6380")
        return

    finance = Finance(heather, dimension=128)

    try:
        run(finance)
    finally:
        heather.close()


if __name__ == "__main__":
    main()
