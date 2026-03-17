"""Interactive terminal for Muse — creative idea blender."""

from .blender import Muse

# ANSI colors
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_blend(result):
    """Print a blend result with colors."""
    print(f"\n{DIM}  inputs: {', '.join(result.inputs)}{RESET}")
    print(f"{DIM}  associations:{RESET}")
    for ing in result.nearby:
        color = GREEN if ing.similarity > 0.8 else YELLOW if ing.similarity > 0.6 else DIM
        print(f"{color}    [{ing.strength} {ing.similarity:.0%}] {ing.text}{RESET}")
    print(f"\n{MAGENTA}{result.idea}{RESET}\n")


def run(muse: Muse):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Muse{RESET} — creative idea blender")
    print(f"{DIM}Commands: add, blend, spark, explore, concepts, stats, quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}muse>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("add "):
            concept = user_input[4:].strip()
            if not concept:
                print(f"{DIM}  usage: add <concept>{RESET}")
                continue
            try:
                muse.add_concept(concept)
                print(f"{DIM}  stored: {concept}{RESET}\n")
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        if user_input.startswith("blend "):
            parts = user_input[6:].strip()
            concepts = [c.strip() for c in parts.split("+") if c.strip()]
            if len(concepts) < 2:
                print(f"{DIM}  usage: blend <concept1> + <concept2> [+ ...]{RESET}")
                continue
            try:
                print(f"{DIM}  blending...{RESET}")
                result = muse.blend(*concepts)
                print_blend(result)
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        if user_input.startswith("spark "):
            concept = user_input[6:].strip()
            if not concept:
                print(f"{DIM}  usage: spark <concept>{RESET}")
                continue
            try:
                print(f"{DIM}  sparking...{RESET}")
                result = muse.spark(concept)
                print_blend(result)
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        if user_input == "explore":
            try:
                print(f"\n{DIM}  exploring concept space...{RESET}\n")
                chain = muse.explore()
                if not chain:
                    print(f"{DIM}  no concepts to explore yet{RESET}\n")
                else:
                    for i, (text, sim) in enumerate(chain, 1):
                        color = GREEN if sim > 0.8 else YELLOW if sim > 0.6 else DIM
                        print(f"{color}  {i}. [{sim:.0%}] {text}{RESET}")
                    print()
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        if user_input == "concepts":
            concepts = muse.embeddings.all_concepts()
            if not concepts:
                print(f"{DIM}  no concepts stored yet{RESET}\n")
            else:
                print(f"\n{DIM}  {len(concepts)} concepts:{RESET}")
                for c in concepts:
                    print(f"{DIM}    - {c}{RESET}")
                print()
            continue

        if user_input == "stats":
            try:
                stats = muse.heather.stats()
                count = muse.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Local cache: {count} concepts{RESET}\n")
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: add, blend, spark, explore, concepts, stats, quit{RESET}\n")
