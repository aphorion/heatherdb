"""Interactive terminal for Genesis — artificial life in memory."""

import time

from .world import World, GenerationReport
from .narrator import Narrator

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Species status colors
STATUS_COLORS = {
    "thriving": GREEN,
    "stable": CYAN,
    "stressed": YELLOW,
    "declining": RED,
    "critical": RED + BOLD,
}

# Simple bar chart characters
BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"


def fitness_bar(fitness: float, width: int = 15) -> str:
    """Render a fitness bar."""
    filled = int(fitness * width)
    bar = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    if fitness > 0.85:
        return f"{GREEN}{bar}{RESET}"
    elif fitness > 0.7:
        return f"{CYAN}{bar}{RESET}"
    elif fitness > 0.55:
        return f"{YELLOW}{bar}{RESET}"
    return f"{RED}{bar}{RESET}"


def print_report(report: GenerationReport):
    """Print a generation report with colors."""
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(
        f"{BOLD}  Generation {report.generation}{RESET}  |  "
        f"pop: {report.population}  |  "
        f"+{report.births} born  -{report.deaths} died  |  "
        f"{len(report.species)} species"
    )
    print(f"{BOLD}{'=' * 60}{RESET}")

    for sp in sorted(report.species, key=lambda s: s.size, reverse=True):
        color = STATUS_COLORS.get(sp.status, DIM)
        bar = fitness_bar(sp.avg_fitness)
        print(
            f"  {color}\u25cf {sp.name:<10}{RESET} "
            f"{sp.size:>3} creatures  "
            f"{bar} {sp.avg_fitness:.2f}  "
            f"{color}{sp.status}{RESET}"
        )

    for event in report.events:
        if "EXTINCTION" in event:
            print(f"\n  {RED}{BOLD}\u2620  {event}{RESET}")
        elif "SPECIATION" in event:
            print(f"\n  {GREEN}{BOLD}\u2728 {event}{RESET}")
        elif "PRESSURE" in event:
            print(f"\n  {YELLOW}\u26a0  {event}{RESET}")

    print()


def run(world: World, narrator: Narrator):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Genesis{RESET} — artificial life in associative memory")
    print(f"{DIM}Commands: step [n], auto, narrate, stats, quit{RESET}")
    print(f"{DIM}The EAM is the universe. Fitness = reconstruction fidelity.{RESET}\n")

    print(f"{DIM}Seeding initial population...{RESET}")
    world.seed()
    print(f"{DIM}Created {len(world.creatures)} creatures in the memory.{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}genesis>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            # Default: step 1
            report = world.step()
            print_report(report)
            continue

        if user_input == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("step"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            for i in range(n):
                report = world.step()
                print_report(report)
            continue

        if user_input == "auto":
            print(f"{DIM}  auto-running (Ctrl+C to stop)...{RESET}\n")
            try:
                while True:
                    report = world.step()
                    print_report(report)

                    # Narrate every 5 generations
                    if report.generation % 5 == 0:
                        print(f"{MAGENTA}{DIM}  [narrator]{RESET}")
                        narration = narrator.narrate(world.history)
                        print(f"  {MAGENTA}{narration}{RESET}\n")

                    # Narrate extinctions immediately
                    for event in report.events:
                        if "EXTINCTION" in event:
                            species_name = event.split(":")[1].strip().split(" ")[0]
                            eulogy = narrator.eulogy(species_name, world.history)
                            print(f"  {DIM}{eulogy}{RESET}\n")

                    time.sleep(1.5)
            except KeyboardInterrupt:
                print(f"\n{DIM}  auto-run stopped{RESET}\n")
            continue

        if user_input == "narrate":
            if not world.history:
                print(f"{DIM}  no history yet — run some generations first{RESET}\n")
                continue
            print(f"\n{MAGENTA}{DIM}  [narrator]{RESET}")
            narration = narrator.narrate(world.history)
            print(f"  {MAGENTA}{narration}{RESET}\n")
            continue

        if user_input == "stats":
            try:
                eam_stats = world.heather.stats()
                print(f"\n{DIM}  Generation: {world.generation}")
                print(f"  Population: {len(world.creatures)}")
                print(f"  Species: {len(world._prev_species)}")
                print(f"  HeatherDB locations: {eam_stats.get('num_locations', '?')}")
                print(f"  Total writes: {eam_stats.get('total_writes', '?')}")
                print(f"  Avg write count: {eam_stats.get('avg_write_count', '?')}")
                print(f"  Max write count: {eam_stats.get('max_write_count', '?')}{RESET}\n")
            except Exception as e:
                print(f"{DIM}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: step [n], auto, narrate, stats, quit{RESET}\n")
