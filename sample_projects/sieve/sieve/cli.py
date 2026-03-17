"""CLI interface for Sieve — near-duplicate detection."""

import csv
import json
import sys
from pathlib import Path

from .detector import Sieve, DuplicateResult

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

VERDICT_COLORS = {
    "DUPLICATE": RED,
    "SIMILAR": YELLOW,
    "UNIQUE": GREEN,
}


def load_texts(filepath: str, field: str = "text") -> list[str]:
    """Load texts from a file (.jsonl, .csv, or .txt)."""
    path = Path(filepath)
    texts = []

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    texts.append(obj[field])

    elif path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if field in row:
                    texts.append(row[field])
                else:
                    # Use first column
                    texts.append(list(row.values())[0])

    else:  # .txt or any other
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    return texts


def print_result(result: DuplicateResult, show_matches: bool = True):
    """Print a single duplicate check result."""
    color = VERDICT_COLORS.get(result.verdict, DIM)
    print(f"  {color}{BOLD}{result.verdict}{RESET} {DIM}(fidelity: {result.fidelity:.2f}){RESET}")
    print(f"  {result.text[:100]}")

    if show_matches and result.matches:
        print(f"{DIM}  matches:{RESET}")
        for m in result.matches:
            print(f"{DIM}    [{m.similarity:.2f}] {m.text[:80]}{RESET}")
    print()


def cmd_ingest(sieve: Sieve, args: list[str]):
    """Ingest records from a file."""
    if not args:
        print(f"{RED}  usage: python run.py ingest <file> [--field <name>]{RESET}")
        sys.exit(1)

    filepath = args[0]
    field = "text"
    if "--field" in args:
        idx = args.index("--field")
        if idx + 1 < len(args):
            field = args[idx + 1]

    print(f"{DIM}Loading records from {filepath}...{RESET}")
    texts = load_texts(filepath, field=field)
    print(f"{DIM}Ingesting {len(texts)} records...{RESET}")
    count = sieve.ingest(texts)
    print(f"{GREEN}  ingested {count} records{RESET}\n")


def cmd_check(sieve: Sieve, args: list[str]):
    """Check a single record for duplicates."""
    if not args:
        print(f"{RED}  usage: python run.py check \"some text here\"{RESET}")
        sys.exit(1)

    text = " ".join(args)
    result = sieve.check(text)
    print()
    print_result(result)


def cmd_scan(sieve: Sieve, args: list[str]):
    """Scan a file for internal duplicates."""
    if not args:
        print(f"{RED}  usage: python run.py scan <file> [--field <name>]{RESET}")
        sys.exit(1)

    filepath = args[0]
    field = "text"
    if "--field" in args:
        idx = args.index("--field")
        if idx + 1 < len(args):
            field = args[idx + 1]

    print(f"{DIM}Loading records from {filepath}...{RESET}")
    texts = load_texts(filepath, field=field)
    print(f"{DIM}Scanning {len(texts)} records for duplicates...{RESET}\n")

    report = sieve.scan(texts)

    # Print each result
    for i, result in enumerate(report.results, 1):
        color = VERDICT_COLORS.get(result.verdict, DIM)
        prefix = f"{color}{BOLD}{result.verdict:>9}{RESET}"
        fid = f"{DIM}({result.fidelity:.2f}){RESET}" if result.fidelity > 0 else f"{DIM}(first){RESET}"
        print(f"  {i:>3}. {prefix} {fid} {result.text[:70]}")

        if result.is_duplicate and result.matches:
            best = result.matches[0]
            print(f"       {DIM}^ matches: {best.text[:60]} [{best.similarity:.2f}]{RESET}")

    # Summary
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}Scan summary:{RESET}")
    print(f"    Total records:  {report.total}")
    print(f"    {RED}Duplicates:     {report.duplicates}{RESET}")
    print(f"    {YELLOW}Similar:        {report.similar}{RESET}")
    print(f"    {GREEN}Unique:         {report.unique}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


def cmd_interactive(sieve: Sieve):
    """Interactive mode: type records, get instant feedback."""
    print(f"\n{BOLD}{MAGENTA}Sieve{RESET} — interactive duplicate checker")
    print(f"{DIM}Type a record to check. Commands: /ingest <file>, /stats, /quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}check>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("/ingest "):
            filepath = user_input[8:].strip()
            try:
                texts = load_texts(filepath)
                count = sieve.ingest(texts)
                print(f"{GREEN}  ingested {count} records{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "/stats":
            try:
                stats = sieve.heather.stats()
                count = sieve.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Local cache: {count} records")
                print(f"  Duplicate threshold: {sieve.duplicate_threshold}")
                print(f"  Similar threshold: {sieve.similar_threshold}{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        # Check the input
        try:
            result = sieve.check(user_input)
            print_result(result)
        except Exception as e:
            print(f"{RED}  error: {e}{RESET}\n")


def cmd_stats(sieve: Sieve):
    """Show stats."""
    stats = sieve.heather.stats()
    count = sieve.embeddings.count()
    print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
          f"{stats.get('total_writes', '?')} total writes")
    print(f"  Local cache: {count} records")
    print(f"  Duplicate threshold: {sieve.duplicate_threshold}")
    print(f"  Similar threshold: {sieve.similar_threshold}{RESET}\n")
