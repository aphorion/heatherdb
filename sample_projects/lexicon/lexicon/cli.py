"""Interactive terminal for Lexicon — word meaning from pure memory."""

from __future__ import annotations

import os
import time

from .engine import Lexicon

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def similarity_color(sim: float) -> str:
    if sim > 0.3:
        return GREEN
    if sim > 0.1:
        return YELLOW
    return DIM


def cmd_ingest(lexicon: Lexicon, path: str):
    """Ingest a text file into the EAM."""
    if not os.path.exists(path):
        print(f"{RED}  file not found: {path}{RESET}\n")
        return

    with open(path, "r") as f:
        text = f.read()

    print(f"\n{BOLD}Ingesting {os.path.basename(path)}...{RESET}")
    start = time.time()

    def progress(sent, total, windows):
        elapsed = time.time() - start
        print(f"  {DIM}{sent}/{total} sentences, {windows} context windows, "
              f"{elapsed:.1f}s{RESET}", end="\r")

    windows = lexicon.ingest_corpus(text, progress_fn=progress)
    elapsed = time.time() - start

    print(f"\n{GREEN}  Done: {lexicon.sentences_processed} sentences, "
          f"{windows} context windows written{RESET}")
    print(f"{DIM}  Vocabulary: {lexicon.vocab.size} words")
    print(f"  Time: {elapsed:.1f}s{RESET}\n")


def cmd_associates(lexicon: Lexicon, word: str):
    """Show words that co-occur with the given word."""
    print(f"\n{BOLD}Associates of '{word}':{RESET}")
    fid = lexicon.fidelity(word)
    print(f"{DIM}  (fidelity: {fid:.3f} — how strongly SDM recognizes this word){RESET}\n")

    results = lexicon.associates(word, top_k=15)
    if not results:
        print(f"  {DIM}no associates found{RESET}\n")
        return

    for w, sim in results:
        color = similarity_color(sim)
        bar_len = int(max(0, sim) * 40)
        bar = "█" * bar_len
        print(f"  {color}{w:20s} {sim:+.4f}  {bar}{RESET}")
    print()


def cmd_similarity(lexicon: Lexicon, word_a: str, word_b: str):
    """Compare two words by their EAM reconstructions."""
    sim = lexicon.similarity(word_a, word_b)
    color = similarity_color(sim)
    print(f"\n  {BOLD}similarity({word_a}, {word_b}){RESET} = {color}{sim:.4f}{RESET}")

    # Also show raw vector similarity (should be ~0 for random vectors)
    import numpy as np
    va = np.array(lexicon.vocab.get(word_a))
    vb = np.array(lexicon.vocab.get(word_b))
    raw = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    print(f"  {DIM}(raw vector similarity: {raw:.4f} — random baseline){RESET}\n")


def cmd_analogy(lexicon: Lexicon, a: str, b: str, c: str):
    """Solve: a is to b as c is to ???"""
    print(f"\n{BOLD}  {a} → {b}  ::  {c} → ???{RESET}\n")

    results = lexicon.analogy(a, b, c, top_k=10)
    for w, sim in results:
        color = similarity_color(sim)
        print(f"  {color}{w:20s} {sim:+.4f}{RESET}")
    print()


def cmd_blend(lexicon: Lexicon, words: list[str]):
    """Find concepts at the intersection of multiple words."""
    print(f"\n{BOLD}Blend: {' + '.join(words)}{RESET}\n")

    results = lexicon.blend(*words, top_k=10)
    for w, sim in results:
        color = similarity_color(sim)
        bar_len = int(max(0, sim) * 40)
        bar = "█" * bar_len
        print(f"  {color}{w:20s} {sim:+.4f}  {bar}{RESET}")
    print()


def cmd_stats(lexicon: Lexicon):
    """Show vocabulary and SDM stats."""
    try:
        stats = lexicon.heather.stats()
        print(f"\n{DIM}  Vocabulary: {lexicon.vocab.size} words")
        print(f"  Context windows written: {lexicon.windows_written}")
        print(f"  Sentences processed: {lexicon.sentences_processed}")
        print(f"  HeatherDB: {stats.get('num_locations', '?')} locations, "
              f"{stats.get('total_writes', '?')} total writes{RESET}\n")
    except Exception as e:
        print(f"{RED}  error: {e}{RESET}\n")


def run(lexicon: Lexicon):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Lexicon{RESET} — word meaning from pure memory")
    print(f"{DIM}Feed text → SDM learns co-occurrence patterns → meaning emerges{RESET}")
    print(f"{DIM}Commands:{RESET}")
    print(f"{DIM}  ingest <file>          Feed a text file into the EAM{RESET}")
    print(f"{DIM}  word <word>            Show associates of a word{RESET}")
    print(f"{DIM}  sim <a> <b>            Compare two words{RESET}")
    print(f"{DIM}  analogy <a> <b> <c>    a→b as c→???{RESET}")
    print(f"{DIM}  blend <w1> <w2> ...    Find the intersection concept{RESET}")
    print(f"{DIM}  stats                  Show vocabulary and SDM stats{RESET}")
    print(f"{DIM}  quit                   Exit{RESET}\n")

    # Auto-ingest sample data if it exists and SDM is empty
    sample = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "sample_data", "corpus.txt")
    if os.path.exists(sample) and lexicon.windows_written == 0:
        print(f"{DIM}Sample corpus found. Ingesting automatically...{RESET}")
        cmd_ingest(lexicon, sample)

    while True:
        try:
            user_input = input(f"{CYAN}lex>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if cmd == "ingest":
            if len(parts) < 2:
                print(f"{DIM}  usage: ingest <filepath>{RESET}\n")
                continue
            try:
                cmd_ingest(lexicon, parts[1])
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "word":
            if len(parts) < 2:
                print(f"{DIM}  usage: word <word>{RESET}\n")
                continue
            try:
                cmd_associates(lexicon, parts[1])
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "sim":
            if len(parts) < 3:
                print(f"{DIM}  usage: sim <word_a> <word_b>{RESET}\n")
                continue
            try:
                cmd_similarity(lexicon, parts[1], parts[2])
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "analogy":
            if len(parts) < 4:
                print(f"{DIM}  usage: analogy <a> <b> <c>   (a→b as c→???){RESET}\n")
                continue
            try:
                cmd_analogy(lexicon, parts[1], parts[2], parts[3])
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "blend":
            if len(parts) < 3:
                print(f"{DIM}  usage: blend <word1> <word2> [word3 ...]{RESET}\n")
                continue
            try:
                cmd_blend(lexicon, parts[1:])
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if cmd == "stats":
            cmd_stats(lexicon)
            continue

        print(f"{DIM}  unknown command. try: ingest, word, sim, analogy, blend, stats, quit{RESET}\n")
