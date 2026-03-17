#!/usr/bin/env python3
"""Generate text through HeatherDB — chunk at a time.

Think then Speak: each generation step produces a full chunk of text
from a single EAM read. The encoder and decoder live in RAM.
The knowledge (EAM) lives on disk in HeatherDB.

One thought = one disk read = one chunk of text.

Usage:
    python generate.py --prompt "ROMEO:" --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from broca.client import HeatherClient
from broca.model import CharLM
from broca.data import CharVocab


def load_model_and_vocab(export_path: str) -> tuple[CharLM, CharVocab]:
    """Reconstruct model and vocabulary from export."""
    export_path = Path(export_path)
    with open(export_path) as f:
        export = json.load(f)

    # Rebuild vocabulary
    vd = export["vocab"]
    vocab = CharVocab.__new__(CharVocab)
    vocab.char_to_idx = vd["char_to_idx"]
    vocab.idx_to_char = {int(k): v for k, v in vd["idx_to_char"].items()}
    vocab.vocab_size = len(vocab.char_to_idx)

    # Rebuild model (no EAM needed — that lives in HeatherDB)
    mc = export["model_config"]
    model = CharLM(
        vocab_size=mc["vocab_size"],
        embed_dim=mc["embed_dim"],
        memory_dim=mc["memory_dim"],
        context_length=mc["context_length"],
        chunk_size=mc.get("chunk_size", 32),
    )

    # Load weights
    ctrl_name = export.get("controller_state")
    weights_path = export_path.parent / ctrl_name if ctrl_name else \
        export_path.with_name(export_path.stem + "_model.pt")

    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.embedding.load_state_dict(weights["embedding"])
    model.encoder_gru.load_state_dict(weights["encoder_gru"])
    model.decoder_gru.load_state_dict(weights["decoder_gru"])
    model.output.load_state_dict(weights["output"])
    model.eval()

    return model, vocab


def generate(
    model: CharLM,
    vocab: CharVocab,
    heather: HeatherClient,
    prompt: str,
    num_chunks: int = 16,
    temperature: float = 0.8,
) -> str:
    """Chunk-level generation with EAM read via HeatherDB.

    One thought per chunk. One HeatherDB read per thought.
    Decoder renders the full chunk in pure RAM.
    """
    context = vocab.encode(prompt)
    generated = list(context)

    for _ in range(num_chunks):
        window = generated[-model.context_length:]
        if len(window) < model.context_length:
            window = [0] * (model.context_length - len(window)) + window

        x = torch.tensor([window], dtype=torch.long)

        with torch.no_grad():
            thought = model.encode(x)

        # One HeatherDB read per thought
        query = thought[0].cpu().numpy().astype(np.float64).tolist()
        reconstruction = np.array(heather.read(query), dtype=np.float64)

        # Decode chunk locally (pure RAM, no disk)
        thought_vector = torch.tensor(
            reconstruction, dtype=torch.float32
        ).unsqueeze(0)
        seed = torch.tensor([window[-1]], dtype=torch.long)

        with torch.no_grad():
            chunk = model.generate_chunk(thought_vector, seed, temperature)

        generated.extend(chunk)

    return vocab.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text via HeatherDB")
    parser.add_argument("--prompt", default="ROMEO:", help="Starting text")
    parser.add_argument("--chunks", type=int, default=16, help="Number of chunks to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    chunk_size_label = ""

    print("=== Broca: Think then Speak via HeatherDB ===\n")

    export_path = Path(__file__).parent / "broca.json"
    if not export_path.exists():
        print("broca.json not found. Run train.py first.")
        sys.exit(1)

    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    collection = os.getenv("COLLECTION", "broca")
    heather = HeatherClient(url=url, collection=collection)

    print("Checking HeatherDB connection...", end=" ", flush=True)
    if not heather.health():
        print("FAILED")
        print("HeatherDB is not running. Start it with:")
        print("  cargo run --release -p heather_server -- --dimension 128 --data-dir ./data")
        sys.exit(1)
    print("OK")

    stats = heather.stats()
    print(f"  Collection '{collection}': {stats['num_locations']} locations, "
          f"{stats['total_writes']:.0f} total writes\n")

    print("Loading model + vocab...", end=" ", flush=True)
    model, vocab = load_model_and_vocab(str(export_path))
    print(f"OK ({vocab.vocab_size} chars, chunk_size={model.chunk_size})")
    print(f"  {args.chunks} chunks = {args.chunks * model.chunk_size} chars "
          f"from {args.chunks} HeatherDB reads\n")

    if args.interactive:
        print("Interactive mode. Type a prompt, press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if not prompt:
                    continue
                text = generate(model, vocab, heather, prompt,
                                num_chunks=args.chunks, temperature=args.temperature)
                print(text)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        print(f"Prompt: {args.prompt!r}\n")
        text = generate(model, vocab, heather, args.prompt,
                        num_chunks=args.chunks, temperature=args.temperature)
        print(text)

    heather.close()


if __name__ == "__main__":
    main()
