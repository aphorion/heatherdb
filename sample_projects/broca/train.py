#!/usr/bin/env python3
"""Train Broca: chunk-level LM with EAM memory.

Think then Speak architecture:
  Phase 1: Train encoder GRU + decoder GRU on chunk prediction (no EAM).
  Phase 2: Freeze controller, encode all training contexts, write to EAM.
  Export: EAM state to JSON + model weights to .pt for HeatherDB import.

Usage:
    python train.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from broca.model import CharLM
from broca.data import download_shakespeare, CharVocab, ChunkDataset

# --- Hyperparameters ---
EMBED_DIM = 64
MEMORY_DIM = 128
CONTEXT_LENGTH = 128
CHUNK_SIZE = 32
NUM_LOCATIONS = 2000
K = 20
BETA = 5.0
T_MAX = 3
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
TRAIN_SPLIT = 0.9
SEED = 42


def export_eam(model: CharLM, vocab: CharVocab, path: str) -> str:
    """Export EAM state + model weights for HeatherDB import."""
    mem = model.memory
    out = Path(path)
    num_locs = mem.addresses.shape[0]

    config = {
        "d": mem.dim,
        "l_0": num_locs,
        "l_max": num_locs * 2,
        "k": mem.k,
        "eta_0": mem.eta,
        "lambda": 0.9999,
        "eta_min": 0.001,
        "tau_split": 0.3,
        "tau_merge": 0.95,
        "gamma": 1.0,
        "tau_damp": 10.0,
        "tau_overload": 100.0,
        "beta": mem.beta,
        "t_max": mem.t_max,
        "epsilon": 1e-6,
    }

    addresses = mem.addresses.detach().cpu().double()
    counters = mem.counters.detach().cpu().double()
    write_counts = mem.write_counts.detach().cpu().double()

    locations = [
        {
            "id": i,
            "address": addresses[i].tolist(),
            "counter": counters[i].tolist(),
            "write_count": float(write_counts[i]),
        }
        for i in range(num_locs)
    ]

    # Save model weights (encoder + decoder + shared embedding)
    model_path = out.with_name(out.stem + "_model.pt")
    torch.save({
        "embedding": model.embedding.state_dict(),
        "encoder_gru": model.encoder_gru.state_dict(),
        "decoder_gru": model.decoder_gru.state_dict(),
        "output": model.output.state_dict(),
    }, model_path)

    export = {
        "config": config,
        "locations": locations,
        "controller_state": model_path.name,
        "prototypes": {},
        "vocab": {
            "char_to_idx": vocab.char_to_idx,
            "idx_to_char": {str(k): v for k, v in vocab.idx_to_char.items()},
        },
        "model_config": {
            "vocab_size": model.vocab_size,
            "embed_dim": model.embed_dim,
            "memory_dim": model.memory_dim,
            "context_length": model.context_length,
            "chunk_size": model.chunk_size,
        },
    }
    out.write_text(json.dumps(export))
    return str(out)


@torch.no_grad()
def generate_sample(
    model: CharLM,
    vocab: CharVocab,
    prompt: str = "ROMEO:",
    num_chunks: int = 10,
    temperature: float = 0.8,
    use_memory: bool = False,
) -> str:
    """Generate text by producing chunks. One thought per chunk."""
    model.eval()
    context = vocab.encode(prompt)
    generated = list(context)

    for _ in range(num_chunks):
        window = generated[-model.context_length:]
        if len(window) < model.context_length:
            window = [0] * (model.context_length - len(window)) + window

        x = torch.tensor([window], dtype=torch.long)
        thought = model.encode(x)

        if use_memory and model.memory.num_written() > 0:
            thought = model.memory.read(thought)

        seed = torch.tensor([window[-1]], dtype=torch.long)
        chunk = model.generate_chunk(thought, seed, temperature)
        generated.extend(chunk)

    return vocab.decode(generated)


def main():
    print("=== Broca: Chunk-Level LM with EAM (Think then Speak) ===\n")
    torch.manual_seed(SEED)

    # Load data
    text = download_shakespeare()
    vocab = CharVocab(text)
    print(f"Corpus: {len(text):,} characters, {vocab.vocab_size} unique chars\n")

    split_idx = int(len(text) * TRAIN_SPLIT)
    train_ds = ChunkDataset(text[:split_idx], vocab, CONTEXT_LENGTH, CHUNK_SIZE)
    val_ds = ChunkDataset(text[split_idx:], vocab, CONTEXT_LENGTH, CHUNK_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds):,} windows ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds):,} windows")
    print(f"Chunk size: {CHUNK_SIZE} chars per thought\n")

    # Build model
    model = CharLM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        memory_dim=MEMORY_DIM,
        context_length=CONTEXT_LENGTH,
        chunk_size=CHUNK_SIZE,
        num_locations=NUM_LOCATIONS,
        k=K, beta=BETA, t_max=T_MAX,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"  Embedding:    {sum(p.numel() for p in model.embedding.parameters()):,}")
    print(f"  Encoder GRU:  {sum(p.numel() for p in model.encoder_gru.parameters()):,}")
    print(f"  Decoder GRU:  {sum(p.numel() for p in model.decoder_gru.parameters()):,}")
    print(f"  Output:       {sum(p.numel() for p in model.output.parameters()):,}")
    print(f"  EAM:          {NUM_LOCATIONS} locations x {MEMORY_DIM}d (buffers)\n")

    # Phase 1: Train encoder + decoder (no EAM)
    print("--- Phase 1: Training encoder + decoder (chunk prediction) ---\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        for context, target_chunk in train_loader:
            optimizer.zero_grad()
            logits = model(context, target_chunk, use_memory=False)
            # logits: [B, chunk_size, vocab_size], target: [B, chunk_size]
            loss = criterion(
                logits.reshape(-1, vocab.vocab_size),
                target_chunk.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * target_chunk.numel()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == target_chunk).sum().item()
            total_count += target_chunk.numel()

        # Validation
        model.eval()
        val_loss, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for context, target_chunk in val_loader:
                logits = model(context, target_chunk, use_memory=False)
                val_loss += criterion(
                    logits.reshape(-1, vocab.vocab_size),
                    target_chunk.reshape(-1),
                ).item() * target_chunk.numel()
                val_correct += (logits.argmax(dim=-1) == target_chunk).sum().item()
                val_count += target_chunk.numel()

        train_ppl = math.exp(total_loss / total_count)
        val_ppl = math.exp(val_loss / val_count)
        train_acc = total_correct / total_count
        val_acc = val_correct / val_count

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}, "
                  f"train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

    print(f"\nFinal: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}\n")

    # Sample without memory (chunk generation)
    print("--- Sample (no memory, chunk generation) ---\n")
    print(generate_sample(model, vocab, use_memory=False))
    print("\n")

    # Phase 2: Populate EAM
    print("--- Phase 2: Populating EAM ---\n")
    model.freeze_controller()
    model.eval()
    model.memory.clear()

    written = 0
    with torch.no_grad():
        for context, _ in train_loader:
            encoded = model.encode(context)
            model.memory.write(encoded)
            written += context.shape[0]

    print(f"  Wrote {written:,} thought vectors to EAM")
    print(f"  Active locations: {model.memory.num_written()}/{NUM_LOCATIONS}\n")

    # Compare direct vs memory path
    print("--- Evaluation: direct vs memory (chunk-level) ---\n")
    model.eval()

    for label, use_mem in [("Direct", False), ("Memory", True)]:
        total_loss, total_correct, total_count = 0.0, 0, 0
        with torch.no_grad():
            for context, target_chunk in val_loader:
                logits = model(context, target_chunk, use_memory=use_mem)
                total_loss += criterion(
                    logits.reshape(-1, vocab.vocab_size),
                    target_chunk.reshape(-1),
                ).item() * target_chunk.numel()
                total_correct += (logits.argmax(dim=-1) == target_chunk).sum().item()
                total_count += target_chunk.numel()
        ppl = math.exp(total_loss / total_count)
        acc = total_correct / total_count
        print(f"  {label:8s}: ppl={ppl:.2f}, acc={acc:.1%}")

    print()

    # Sample with memory (chunk generation)
    print("--- Sample (with EAM, chunk generation) ---\n")
    print(generate_sample(model, vocab, use_memory=True))
    print("\n")

    # Export
    out_path = Path(__file__).parent / "broca.json"
    print(f"Exporting to {out_path}...")
    export_eam(model, vocab, str(out_path))
    print(f"  Created: broca.json + broca_model.pt")
    print(f"  Locations: {model.memory.num_locations}")
    print(f"  Vocab: {vocab.vocab_size} chars")
    print(f"  Chunk size: {CHUNK_SIZE} chars per thought")
    print(f"\nNext steps:")
    print(f"  1. cargo run -p heather_fornix -- import -f {out_path} -c broca -d ./data")
    print(f"  2. cargo run -p heather_server -- --dimension {MEMORY_DIM} --data-dir ./data")
    print(f"  3. python generate.py --interactive")


if __name__ == "__main__":
    main()
