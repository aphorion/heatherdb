"""Cipher CLI — interactive code pattern completion."""

from __future__ import annotations

import json
import os
import textwrap

from .client import HeatherClient
from .embeddings import EmbeddingStore
from .engine import Cipher


def truncate_code(code: str, max_lines: int = 6) -> str:
    lines = code.strip().split("\n")
    if len(lines) <= max_lines:
        return code.strip()
    return "\n".join(lines[:max_lines]) + f"\n  ... ({len(lines) - max_lines} more lines)"


def print_results(label: str, results: list[tuple[str, str, float]]):
    if not results:
        print(f"  (no results)")
        return
    for i, (desc, code, sim) in enumerate(results, 1):
        bar_len = int(sim * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {i}. [{bar}] {sim:.3f}  {desc}")
        print(textwrap.indent(truncate_code(code), "     "))
        print()


def cmd_query(cipher: Cipher, args: str):
    if not args:
        print("Usage: query <description>")
        return
    result = cipher.query(args, k=5)
    print(f"\nQuery: \"{result.query}\"")
    print(f"Fidelity: {result.fidelity:.4f}\n")

    print("── Vector DB (direct nearest) ──")
    print_results("direct", result.nearest[:5])

    print("── SDM Reconstruction (pattern blend) ──")
    print_results("blend", result.blend_nearest[:5])

    # Show what's different
    direct_descs = {d for d, _, _ in result.nearest[:5]}
    blend_descs = {d for d, _, _ in result.blend_nearest[:5]}
    only_blend = blend_descs - direct_descs
    if only_blend:
        print(f"  SDM surfaced {len(only_blend)} snippet(s) not in direct top-5:")
        for d in only_blend:
            print(f"    → {d}")
    print()


def cmd_blend(cipher: Cipher, args: str):
    parts = [p.strip().strip('"').strip("'") for p in args.split("+")]
    if len(parts) < 2:
        print('Usage: blend "concept1" + "concept2" [+ ...]')
        print('Example: blend HTTP request + error handling')
        return

    result = cipher.blend(*parts, k=5)
    print(f"\nBlending: {' ⊕ '.join(result.concepts)}")
    print(f"Blend fidelity: {result.fidelity:.4f}\n")

    print("── Nearest to blend intersection ──")
    print_results("blend", result.nearest)


def cmd_complete(cipher: Cipher, args: str):
    if not args:
        print("Usage: complete <partial code or description>")
        print("Example: complete import json\\nwith open")
        return

    # Unescape literal \\n
    code = args.replace("\\n", "\n")
    result = cipher.complete(code, k=5)
    print(f"\nPartial code fidelity: {result.fidelity:.4f}\n")

    print("── SDM suggests these patterns ──")
    print_results("completion", result.blend_nearest)


def cmd_ingest(cipher: Cipher, args: str):
    if not args:
        print("Usage: ingest <path-to-json>")
        return
    path = args.strip()
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path) as f:
        snippets = json.load(f)
    cipher.ingest_batch(snippets)
    print(f"Ingested {len(snippets)} snippets (total: {cipher.snippets_stored})")


def cmd_stats(cipher: Cipher, _args: str):
    try:
        stats = cipher.heather.stats()
        print(f"\nHeatherDB:")
        print(f"  Hard locations: {stats.get('hard_location_count', '?')}")
        print(f"  Dimension: {stats.get('dimension', '?')}")
        print(f"  Writes: {stats.get('total_writes', '?')}")
    except Exception as e:
        print(f"  HeatherDB error: {e}")

    print(f"\nCipher:")
    print(f"  Snippets stored: {cipher.snippets_stored}")
    print(f"  Embedding dim: {cipher.embeddings.dimension}")
    print(f"  DB entries: {cipher.embeddings.count()}")
    print()


def run_cli():
    url = os.getenv("HEATHER_URL", "http://localhost:6380")
    db_path = os.getenv("CIPHER_DB", "cipher.db")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")

    print("=" * 60)
    print("  CIPHER — Code Pattern Completion via EAM")
    print("=" * 60)
    print()

    heather = HeatherClient(url)
    if not heather.health():
        print(f"WARNING: HeatherDB not reachable at {url}")
        print("Start it with: cargo run --release -p heather_server -- --dimension 384")
        print()

    embeddings = EmbeddingStore(db_path)
    cipher = Cipher(heather, embeddings)

    # Auto-ingest sample data if DB is empty
    if embeddings.count() == 0:
        snippets_path = os.path.join(data_dir, "snippets.json")
        if os.path.exists(snippets_path):
            print("Loading sample snippets...")
            with open(snippets_path) as f:
                snippets = json.load(f)
            cipher.ingest_batch(snippets)
            print(f"Ingested {len(snippets)} snippets.\n")

    commands = {
        "query": cmd_query,
        "blend": cmd_blend,
        "complete": cmd_complete,
        "ingest": cmd_ingest,
        "stats": cmd_stats,
    }

    print("Commands:")
    print("  query <description>          — find code patterns")
    print("  blend <concept1> + <concept2> — find intersection")
    print("  complete <partial code>       — complete a pattern")
    print("  ingest <file.json>            — add more snippets")
    print("  stats                         — show stats")
    print("  quit                          — exit")
    print()

    while True:
        try:
            line = input("cipher> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        if line in ("quit", "exit", "q"):
            print("Bye!")
            break

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in commands:
            try:
                commands[cmd](cipher, args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: query, blend, complete, ingest, stats, quit")
