"""CLI interface for Sentinel — anomaly detection with reconstruction diff."""

import sys
from pathlib import Path

from .detector import Sentinel, Anomaly

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

LEVEL_COLORS = {
    "NORMAL": GREEN,
    "SUSPICIOUS": YELLOW,
    "ANOMALY": RED,
}

LEVEL_ICONS = {
    "NORMAL": "\u2713",      # checkmark
    "SUSPICIOUS": "\u26a0",  # warning
    "ANOMALY": "\u2620",     # skull
}


def load_lines(filepath: str) -> list[str]:
    """Load non-empty lines from a text file."""
    lines = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def print_anomaly(result: Anomaly, verbose: bool = True):
    """Print a single anomaly result."""
    color = LEVEL_COLORS.get(result.level, DIM)
    icon = LEVEL_ICONS.get(result.level, "?")

    print(f"  {color}{icon} {BOLD}{result.level}{RESET} {DIM}(fidelity: {result.fidelity:.2f}){RESET}")
    print(f"    input:    {result.text[:90]}")

    if verbose and result.level != "NORMAL":
        print(f"    {CYAN}expected: {result.expected[:90]}{RESET}")
        if result.nearest_input and result.nearest_input != "(no baseline)":
            print(f"    {DIM}closest:  {result.nearest_input[:90]} [{result.nearest_input_similarity:.2f}]{RESET}")
    print()


def cmd_train(sentinel: Sentinel, args: list[str]):
    """Train on normal log data."""
    if not args:
        print(f"{RED}  usage: python run.py train <file>{RESET}")
        sys.exit(1)

    filepath = args[0]
    print(f"{DIM}Loading normal patterns from {filepath}...{RESET}")
    texts = load_lines(filepath)
    print(f"{DIM}Training on {len(texts)} records...{RESET}")

    report = sentinel.train(texts)
    print(f"{GREEN}  trained on {report.count} records{RESET}")
    print(f"{DIM}  baseline fidelity: {report.baseline_fidelity:.2f}{RESET}\n")


def cmd_check(sentinel: Sentinel, args: list[str]):
    """Check a single log line."""
    if not args:
        print(f"{RED}  usage: python run.py check \"log line here\"{RESET}")
        sys.exit(1)

    text = " ".join(args)
    result = sentinel.check(text)
    print()
    print_anomaly(result)


def cmd_monitor(sentinel: Sentinel, args: list[str]):
    """Monitor a file of log lines."""
    if not args:
        print(f"{RED}  usage: python run.py monitor <file>{RESET}")
        sys.exit(1)

    filepath = args[0]
    print(f"{DIM}Loading logs from {filepath}...{RESET}")
    texts = load_lines(filepath)
    print(f"{DIM}Analyzing {len(texts)} log entries...{RESET}\n")

    report = sentinel.monitor(texts)

    for result in report.results:
        color = LEVEL_COLORS.get(result.level, DIM)
        icon = LEVEL_ICONS.get(result.level, "?")

        # Compact line for normal, detailed for anomalies
        if result.level == "NORMAL":
            print(f"  {color}{icon}{RESET} {DIM}{result.text[:80]} ({result.fidelity:.2f}){RESET}")
        else:
            print(f"  {color}{icon} {BOLD}{result.level}{RESET} {DIM}({result.fidelity:.2f}){RESET} {result.text[:70]}")
            print(f"    {CYAN}\u2192 expected: {result.expected[:75]}{RESET}")

    # Summary
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}Monitor summary:{RESET}  {report.total} entries analyzed")
    print(f"    {GREEN}\u2713 Normal:     {report.normal}{RESET}")
    print(f"    {YELLOW}\u26a0 Suspicious: {report.suspicious}{RESET}")
    print(f"    {RED}\u2620 Anomalies:  {report.anomalies}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


def cmd_interactive(sentinel: Sentinel):
    """Interactive mode: paste log lines, get instant analysis."""
    print(f"\n{BOLD}{MAGENTA}Sentinel{RESET} — anomaly detection with reconstruction diff")
    print(f"{DIM}Paste log lines to analyze. Commands: /train <file>, /stats, /quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("/train "):
            filepath = user_input[7:].strip()
            try:
                texts = load_lines(filepath)
                report = sentinel.train(texts)
                print(f"{GREEN}  trained on {report.count} records (baseline fidelity: {report.baseline_fidelity:.2f}){RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "/stats":
            try:
                stats = sentinel.heather.stats()
                count = sentinel.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Baseline records: {count}")
                print(f"  Anomaly threshold: < {sentinel.anomaly_threshold}")
                print(f"  Suspicious threshold: < {sentinel.suspicious_threshold}{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        # Check the log line
        try:
            result = sentinel.check(user_input)
            print_anomaly(result)
        except Exception as e:
            print(f"{RED}  error: {e}{RESET}\n")


def cmd_stats(sentinel: Sentinel):
    """Show stats."""
    stats = sentinel.heather.stats()
    count = sentinel.embeddings.count()
    print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
          f"{stats.get('total_writes', '?')} total writes")
    print(f"  Baseline records: {count}")
    print(f"  Anomaly threshold: < {sentinel.anomaly_threshold}")
    print(f"  Suspicious threshold: < {sentinel.suspicious_threshold}{RESET}\n")
