"""Interactive terminal for Whisper — lossy semantic compression."""

import sys
from pathlib import Path

from .memory import Whisper, Reconstruction

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_reconstruction(recon: Reconstruction):
    """Print a reconstruction result."""
    print(f"\n{DIM}  fidelity: {recon.fidelity:.2f}{RESET}")
    print(f"{DIM}  fragments the memory surfaced:{RESET}")
    for f in recon.fragments:
        if f.similarity > 0.8:
            color = GREEN
            label = "vivid"
        elif f.similarity > 0.6:
            color = YELLOW
            label = "clear"
        else:
            color = DIM
            label = "faded"
        print(f"{color}    [{label} {f.similarity:.0%}] {f.text[:90]}{RESET}")

    print(f"\n{MAGENTA}  {recon.interpretation}{RESET}\n")


def cmd_load(whisper: Whisper, args: list[str]):
    """Load a document into memory."""
    if not args:
        print(f"{RED}  usage: python run.py load <file>{RESET}")
        sys.exit(1)

    filepath = args[0]
    if not Path(filepath).exists():
        print(f"{RED}  file not found: {filepath}{RESET}")
        sys.exit(1)

    print(f"{DIM}Loading {filepath}...{RESET}")
    count = whisper.absorb_file(filepath)
    print(f"{GREEN}  absorbed {count} sentences into shared memory{RESET}\n")


def cmd_ask(whisper: Whisper, args: list[str]):
    """Query the compressed memory."""
    if not args:
        print(f"{RED}  usage: python run.py ask \"your question\"{RESET}")
        sys.exit(1)

    query = " ".join(args)
    recon = whisper.recall(query)
    print_reconstruction(recon)


def cmd_interactive(whisper: Whisper):
    """Interactive mode."""
    print(f"\n{BOLD}{MAGENTA}Whisper{RESET} — lossy semantic compression")
    print(f"{DIM}Documents compressed into EAM. Query from any angle.{RESET}")
    print(f"{DIM}Commands: /load <file>, /sources, /stats, /quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}recall>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("/load "):
            filepath = user_input[6:].strip()
            try:
                if not Path(filepath).exists():
                    print(f"{RED}  file not found: {filepath}{RESET}\n")
                    continue
                print(f"{DIM}  loading {filepath}...{RESET}")
                count = whisper.absorb_file(filepath)
                print(f"{GREEN}  absorbed {count} sentences{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "/sources":
            sources = whisper.embeddings.sources()
            if not sources:
                print(f"{DIM}  no documents loaded{RESET}\n")
            else:
                print(f"\n{DIM}  absorbed documents:{RESET}")
                for source, count in sources:
                    print(f"{DIM}    {source}: {count} sentences{RESET}")
                print()
            continue

        if user_input == "/stats":
            try:
                stats = whisper.heather.stats()
                total = whisper.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Total sentences in memory: {total}{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        # Treat any other input as a query
        try:
            recon = whisper.recall(user_input)
            print_reconstruction(recon)
        except Exception as e:
            print(f"{RED}  error: {e}{RESET}\n")


def cmd_stats(whisper: Whisper):
    """Show stats."""
    stats = whisper.heather.stats()
    total = whisper.embeddings.count()
    sources = whisper.embeddings.sources()
    print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
          f"{stats.get('total_writes', '?')} total writes")
    print(f"  Total sentences: {total}")
    for source, count in sources:
        print(f"    {source}: {count}")
    print(f"{RESET}\n")
