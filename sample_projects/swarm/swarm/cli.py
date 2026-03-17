"""Interactive terminal for Swarm — emergent intelligence from shared memory."""

from .swarm import Swarm, SwarmResult
from .agents import AgentConfig

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"
WHITE = "\033[37m"

AGENT_COLORS = {
    "researcher": CYAN,
    "critic": YELLOW,
    "visionary": MAGENTA,
    "pragmatist": GREEN,
}


def progress_callback(event: str, agent: AgentConfig | None, count: int = 0):
    """Show progress as agents think."""
    if event == "thinking" and agent:
        color = AGENT_COLORS.get(agent.name, DIM)
        print(f"  {color}\u25cb {agent.name}{RESET} {DIM}({agent.role}) thinking...{RESET}", end="", flush=True)
    elif event == "done" and agent:
        print(f" {DIM}{count} thoughts{RESET}")
    elif event == "emerging":
        print(f"\n  {WHITE}{BOLD}\u2727 extracting emergent patterns from shared memory...{RESET}")
    elif event == "synthesizing":
        print(f"  {WHITE}{BOLD}\u2727 synthesizing...{RESET}\n")


def print_result(result: SwarmResult):
    """Print a full swarm result."""
    # Show agent contributions (condensed)
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}Agent Contributions{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    for contrib in result.contributions:
        color = AGENT_COLORS.get(contrib.agent, DIM)
        print(f"\n  {color}{BOLD}\u25cf {contrib.agent}{RESET} {DIM}({len(contrib.thoughts)} thoughts){RESET}")
        for thought in contrib.thoughts[:3]:
            print(f"  {DIM}  {thought[:85]}{RESET}")
        if len(contrib.thoughts) > 3:
            print(f"  {DIM}  ... +{len(contrib.thoughts) - 3} more{RESET}")

    # Show emergent patterns
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  {WHITE}{BOLD}\u2727 Emergent Patterns{RESET} {DIM}(reconstructed from interference){RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")
    for insight in result.emergent:
        color = AGENT_COLORS.get(insight.source_agent, DIM)
        strength = "strong" if insight.similarity > 0.8 else "moderate" if insight.similarity > 0.6 else "faint"
        print(f"  {color}[{insight.source_agent} | {strength} {insight.similarity:.0%}]{RESET}")
        print(f"  {insight.text[:100]}")
        print()

    # Show synthesis
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {WHITE}{BOLD}\u2727 Synthesis{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")
    print(f"  {result.synthesis}\n")


def run(swarm: Swarm):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Swarm{RESET} — emergent intelligence from shared memory")
    print(f"{DIM}4 agents think through one SDM. Insights emerge from interference.{RESET}")
    print(f"{DIM}Commands: think <question>, ask <question>, agents, stats, quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}swarm>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("think "):
            question = user_input[6:].strip()
            if not question:
                print(f"{DIM}  usage: think <question>{RESET}\n")
                continue
            try:
                print(f"\n{DIM}  question: {question}{RESET}\n")
                result = swarm.think(question, verbose_callback=progress_callback)
                print_result(result)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input.startswith("ask "):
            question = user_input[4:].strip()
            if not question:
                print(f"{DIM}  usage: ask <question>{RESET}\n")
                continue
            if swarm.embeddings.count() == 0:
                print(f"{DIM}  no thoughts in memory yet — use 'think' first{RESET}\n")
                continue
            try:
                print(f"\n{DIM}  querying shared memory...{RESET}\n")
                emergent, synthesis = swarm.query(question)

                print(f"  {WHITE}{BOLD}\u2727 Emergent Patterns:{RESET}\n")
                for insight in emergent:
                    color = AGENT_COLORS.get(insight.source_agent, DIM)
                    print(f"  {color}[{insight.source_agent} {insight.similarity:.0%}]{RESET} {insight.text[:90]}")
                print(f"\n  {synthesis}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "agents":
            print(f"\n{DIM}  4 agents sharing one SDM:{RESET}")
            for name, color in AGENT_COLORS.items():
                from .agents import AGENTS
                config = next(a for a in AGENTS if a.name == name)
                print(f"  {color}\u25cf {name:<12}{RESET} {DIM}{config.role}{RESET}")
            print()
            continue

        if user_input == "stats":
            try:
                stats = swarm.heather.stats()
                counts = swarm.embeddings.count_by_agent()
                total = swarm.embeddings.count()
                print(f"\n{DIM}  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes")
                print(f"  Shared memory: {total} thoughts total")
                for agent, count in sorted(counts.items()):
                    color = AGENT_COLORS.get(agent, DIM)
                    print(f"    {color}\u25cf {agent}: {count} thoughts{RESET}")
                print(f"{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: think <question>, ask <question>, agents, stats, quit{RESET}\n")
