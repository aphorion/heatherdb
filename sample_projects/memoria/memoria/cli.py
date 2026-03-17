"""Interactive terminal chat loop for Memoria."""

from .agent import Memoria


# ANSI colors
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_memories(memories):
    """Print recalled memories in dimmed text."""
    if not memories:
        return
    print(f"{DIM}  recalled memories:{RESET}")
    for mem in memories:
        color = GREEN if mem.confidence > 0.85 else YELLOW if mem.confidence > 0.7 else DIM
        print(f"{color}  [{mem.label} {mem.confidence:.0%}] {mem.text[:100]}{RESET}")
    print()


def run(agent: Memoria):
    """Main chat loop."""
    print(f"\n{BOLD}{MAGENTA}Memoria{RESET} — AI with associative memory")
    print(f"{DIM}Type to chat. Commands: /memories, /dream, /quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input == "/memories":
            try:
                stats = agent.heather.stats()
                count = agent.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Local cache: {count} memories{RESET}\n")
            except Exception as e:
                print(f"{DIM}  error fetching stats: {e}{RESET}\n")
            continue

        if user_input == "/dream":
            print(f"\n{MAGENTA}{DIM}  entering dream mode...{RESET}\n")
            try:
                reflection, chain = agent.dream()
                if chain:
                    print(f"{DIM}  association chain:{RESET}")
                    for i, text in enumerate(chain, 1):
                        print(f"{DIM}  {i}. {text[:100]}{RESET}")
                    print()
                print(f"{MAGENTA}{reflection}{RESET}\n")
            except Exception as e:
                print(f"{DIM}  dream failed: {e}{RESET}\n")
            continue

        # Regular chat
        try:
            response, memories = agent.chat(user_input)
            print_memories(memories)
            print(f"{GREEN}memoria>{RESET} {response}\n")
        except Exception as e:
            print(f"{DIM}  error: {e}{RESET}\n")
