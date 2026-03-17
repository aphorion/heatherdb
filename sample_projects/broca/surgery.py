#!/usr/bin/env python3
"""Knowledge Surgery — generate through algebra'd EAM collections.

Requires: broca_shakespeare and broca_bible collections in HeatherDB.
Run the algebra commands first (see below), then this script.

Setup:
    heather-fornix import -f broca.json       -c broca_shakespeare -d ./data
    heather-fornix import -f broca_bible.json -c broca_bible       -d ./data

    # Start server
    cargo run --release -p heather_server -- --dimension 128 --data-dir ./data

    # Algebra
    curl -X POST localhost:6380/algebra/add   -H 'Content-Type: application/json' \
         -d '{"source_a":"broca_shakespeare","source_b":"broca_bible","target":"broca_blend"}'
    curl -X POST localhost:6380/algebra/sub   -H 'Content-Type: application/json' \
         -d '{"source_a":"broca_blend","source_b":"broca_bible","target":"broca_distilled"}'
    curl -X POST localhost:6380/algebra/scale -H 'Content-Type: application/json' \
         -d '{"source":"broca_bible","target":"broca_whisper","alpha":0.3}'

Usage:
    python surgery.py
"""

import json
import os
from pathlib import Path

import httpx
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from broca.model import CharLM
from broca.data import CharVocab


HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")

COLLECTIONS = [
    ("broca_shakespeare", "Pure Shakespeare"),
    ("broca_bible", "Pure Bible"),
    ("broca_blend", "Shakespeare + Bible (add)"),
    ("broca_distilled", "Blend - Bible (sub)"),
    ("broca_whisper", "Bible x 0.3 (scale)"),
]

PROMPTS = [
    "ROMEO:",
    "And God said",
    "To be, or not to be",
    "In the beginning",
]


def load_model(export_path):
    with open(export_path) as f:
        export = json.load(f)
    vd = export["vocab"]
    vocab = CharVocab.__new__(CharVocab)
    vocab.char_to_idx = vd["char_to_idx"]
    vocab.idx_to_char = {int(k): v for k, v in vd["idx_to_char"].items()}
    vocab.vocab_size = len(vocab.char_to_idx)
    mc = export["model_config"]
    model = CharLM(
        vocab_size=mc["vocab_size"], embed_dim=mc["embed_dim"],
        memory_dim=mc["memory_dim"], context_length=mc["context_length"],
        chunk_size=mc.get("chunk_size", 32),
    )
    ctrl = export.get("controller_state")
    weights_path = export_path.parent / ctrl if ctrl else \
        export_path.with_name(export_path.stem + "_model.pt")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.embedding.load_state_dict(weights["embedding"])
    model.encoder_gru.load_state_dict(weights["encoder_gru"])
    model.decoder_gru.load_state_dict(weights["decoder_gru"])
    model.output.load_state_dict(weights["output"])
    model.eval()
    return model, vocab


def generate_through_heather(model, vocab, collection, prompt, num_chunks=10,
                              temperature=0.8):
    context = vocab.encode(prompt)
    generated = list(context)
    client = httpx.Client(base_url=HEATHER_URL, timeout=10)

    for _ in range(num_chunks):
        window = generated[-model.context_length:]
        if len(window) < model.context_length:
            window = [0] * (model.context_length - len(window)) + window

        x = torch.tensor([window], dtype=torch.long)
        with torch.no_grad():
            thought = model.encode(x)

        query = thought[0].numpy().astype(np.float64).tolist()
        resp = client.post(f"/collections/{collection}/read",
                           json={"query": query, "strategy": "iterative"})
        result = np.array(resp.json()["result"], dtype=np.float64)
        thought_vec = torch.tensor(result, dtype=torch.float32).unsqueeze(0)

        seed = torch.tensor([window[-1]], dtype=torch.long)
        with torch.no_grad():
            chunk = model.generate_chunk(thought_vec, seed, temperature)
        generated.extend(chunk)

    client.close()
    return vocab.decode(generated)


def main():
    print("=" * 70)
    print("  KNOWLEDGE SURGERY — Same decoder, different knowledge")
    print("=" * 70)

    # Load Shakespeare model (the decoder we'll use for everything)
    shakespeare_path = Path(__file__).parent / "broca.json"
    if not shakespeare_path.exists():
        print("broca.json not found. Run train.py or download from Colab first.")
        return

    print("\nLoading Shakespeare encoder/decoder...", end=" ", flush=True)
    model, vocab_shk = load_model(shakespeare_path)
    print(f"OK ({vocab_shk.vocab_size} chars)")

    # Also load Bible vocab for Bible-only prompts
    bible_path = Path(__file__).parent / "broca_bible.json"
    vocab_bible = None
    if bible_path.exists():
        with open(bible_path) as f:
            bvd = json.load(f)["vocab"]
        vocab_bible = CharVocab.__new__(CharVocab)
        vocab_bible.char_to_idx = bvd["char_to_idx"]
        vocab_bible.idx_to_char = {int(k): v for k, v in bvd["idx_to_char"].items()}
        vocab_bible.vocab_size = len(vocab_bible.char_to_idx)
        print(f"Loaded Bible vocab ({vocab_bible.vocab_size} chars)")

    # Check HeatherDB
    print("\nChecking HeatherDB...", end=" ", flush=True)
    try:
        resp = httpx.get(f"{HEATHER_URL}/health", timeout=5)
        print("OK")
    except Exception:
        print("FAILED — start HeatherDB first")
        return

    # Check which collections exist
    resp = httpx.get(f"{HEATHER_URL}/collections", timeout=5)
    available = resp.json()["collections"]
    print(f"Available collections: {', '.join(available)}\n")

    # Generate through each collection
    for prompt in PROMPTS:
        print("=" * 70)
        print(f"  PROMPT: {prompt!r}")
        print("=" * 70)

        for collection, label in COLLECTIONS:
            if collection not in available:
                print(f"\n  [{label}] — collection '{collection}' not found, skipping")
                continue

            # Use Shakespeare vocab for Shakespeare-based collections,
            # Bible vocab for pure Bible if available
            if collection == "broca_bible" and vocab_bible:
                v = vocab_bible
            else:
                v = vocab_shk

            # Skip if prompt chars aren't in vocab
            if not all(ch in v.char_to_idx for ch in prompt):
                print(f"\n  [{label}] — prompt contains chars not in vocab, skipping")
                continue

            try:
                text = generate_through_heather(model, v, collection, prompt,
                                                 num_chunks=10, temperature=0.8)
                print(f"\n  [{label}]")
                # Indent output
                for line in text.split("\n"):
                    print(f"    {line}")
            except Exception as e:
                print(f"\n  [{label}] — error: {e}")

        print()

    print("=" * 70)
    print("  Done. Same 150K param decoder. Five different knowledge bases.")
    print("=" * 70)


if __name__ == "__main__":
    main()
