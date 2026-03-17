"""Entry point for Lexicon — word meaning from pure memory."""

import os
from dotenv import load_dotenv

load_dotenv()

from lexicon.client import HeatherClient
from lexicon.vocabulary import Vocabulary
from lexicon.engine import Lexicon
from lexicon.cli import run


def main():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    heather = HeatherClient(url)

    if not heather.health():
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- "
              "--data-dir /tmp/lexicon_db --dimension 128 --port 6380")
        return

    vocab = Vocabulary(dimension=128, cache_path="vocab.json")
    lexicon = Lexicon(heather, vocab, window_size=5)

    try:
        run(lexicon)
    finally:
        vocab.save()
        heather.close()


if __name__ == "__main__":
    main()
