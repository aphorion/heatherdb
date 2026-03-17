"""Interactive terminal for Navigator — one-shot RL through danger memory.

The robot moves forward by default. When it crashes, it remembers the
sensor pattern. Next time it feels something similar, it turns away.
Watch a survival instinct emerge from pure memory.
"""

import time
import random

from .world import (
    Grid, Robot, MAZE_TRAIN, MAZE_TEST, ACTIONS,
    HEADING_NAMES, render, find_open_position,
)
from .brain import Brain

# ANSI
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"


def danger_bar(level: float) -> str:
    """Visual danger meter."""
    filled = int(level * 20)
    bar = "█" * filled + "░" * (20 - filled)
    if level > 0.85:
        return f"{RED}{bar}{RESET}"
    elif level > 0.5:
        return f"{YELLOW}{bar}{RESET}"
    return f"{GREEN}{bar}{RESET}"


def cmd_live(brain: Brain, steps: int = 300, maze_lines=None,
             label: str = "maze", learn: bool = True):
    """Live exploration with danger learning."""
    grid = Grid(maze_lines or MAZE_TRAIN)
    x, y = find_open_position(grid)
    robot = Robot(grid, x, y, heading=random.randint(0, 3))
    trail = []
    danger_batch = []

    # Track learning
    window = 50
    recent_collisions = []

    print(f"\n{BOLD}{'Live learning' if learn else 'Pure recall'} in {label}...{RESET}")
    if learn:
        print(f"{DIM}Robot moves forward by default. Crashes become danger memories.{RESET}\n")
    else:
        print(f"{DIM}Robot navigates using only danger memories from before.{RESET}\n")

    for i in range(steps):
        sensors = robot.sense_relative()

        # Decide based on danger intuition
        action, danger = brain.decide(sensors)

        # Execute
        old_pos = (robot.x, robot.y)
        success = robot.act(action)
        moved = (robot.x, robot.y) != old_pos
        collided = action == "forward" and not moved

        trail.append((robot.x, robot.y))
        brain.update_position(robot.x, robot.y)

        # Learn from crashes
        remembered = False
        if learn and collided:
            danger_batch.append(sensors)
            remembered = True

        recent_collisions.append(1 if collided else 0)

        # Batch write every 10 crashes
        if len(danger_batch) >= 10:
            brain.remember_dangers(danger_batch)
            danger_batch = []

        # Visualize
        if i % 3 == 0 or i == steps - 1:
            print(f"{CLEAR_SCREEN}")
            print(f"  {BOLD}{'Live learning' if learn else 'Transfer test'}"
                  f" — {label} — step {i+1}/{steps}{RESET}")
            print(f"  {DIM}{brain.memories} danger memories{RESET}")

            # Rolling collision rate
            if len(recent_collisions) > window:
                rate = sum(recent_collisions[-window:]) / window * 100
                color = GREEN if rate < 5 else YELLOW if rate < 15 else RED
                print(f"  {DIM}collisions (last {window}): {color}{rate:.0f}%{RESET}")

            print()
            print(render(grid, robot, trail))
            print()

            # Status
            print(f"  {DIM}pos: ({robot.x},{robot.y}) heading: {HEADING_NAMES[robot.heading]}  "
                  f"step: {robot.steps}  collisions: {robot.collisions}{RESET}")

            # Danger + frustration meters
            action_color = GREEN if action == "forward" else YELLOW
            mem_str = f"  {RED}+danger{RESET}" if remembered else ""
            frust = brain.frustration
            frust_str = f"  {MAGENTA}frustration: {frust:.1f}{RESET}" if frust > 0.3 else ""
            print(f"  danger: {danger_bar(danger)} {danger:.2f}  "
                  f"→ {action_color}{action}{RESET}{mem_str}{frust_str}")

            # Sensors
            labels = ["ahead", "a-right", "right", "b-right",
                      "behind", "b-left", "left", "a-left"]
            sensor_str = "  "
            for lbl, val in zip(labels, sensors):
                color = RED if val < 0.2 else YELLOW if val < 0.4 else DIM
                sensor_str += f"{color}{lbl}={val:.1f}{RESET} "
            print(sensor_str)

            time.sleep(0.06)

    # Flush remaining
    if danger_batch:
        brain.remember_dangers(danger_batch)

    # Summary
    collision_rate = robot.collisions / robot.steps * 100 if robot.steps > 0 else 0
    unique_cells = len(set(trail))

    half = len(recent_collisions) // 2
    early = recent_collisions[:half]
    late = recent_collisions[half:]
    early_rate = sum(early) / len(early) * 100 if early else 0
    late_rate = sum(late) / len(late) * 100 if late else 0

    print(f"\n{GREEN}  Done: {steps} steps, {robot.collisions} collisions ({collision_rate:.1f}%){RESET}")
    print(f"{DIM}  Unique cells visited: {unique_cells}")
    print(f"  Danger memories: {brain.memories}")
    print(f"  First half collisions:  {early_rate:.1f}%")
    print(f"  Second half collisions: {late_rate:.1f}%{RESET}")

    improvement = early_rate - late_rate
    if improvement > 2:
        print(f"{GREEN}  Learned: {improvement:.0f}% fewer collisions in second half{RESET}\n")
    elif late_rate < 5:
        print(f"{GREEN}  Navigating safely{RESET}\n")
    else:
        print()


def cmd_transfer(brain: Brain, steps: int = 200):
    """Transfer: new maze, no new learning, pure recall."""
    cmd_live(brain, steps=steps, maze_lines=MAZE_TEST,
             label="new maze", learn=False)


def cmd_compare(brain: Brain, steps: int = 200):
    """Compare EAM vs random in the test maze."""
    grid = Grid(MAZE_TEST)
    x, y = find_open_position(grid)
    heading = random.randint(0, 3)

    # SDM run
    robot_eam = Robot(grid, x, y, heading=heading)
    for _ in range(steps):
        sensors = robot_eam.sense_relative()
        action, _ = brain.decide(sensors)
        robot_eam.act(action)

    # Forward-only run (naive explorer)
    robot_fwd = Robot(grid, x, y, heading=heading)
    for _ in range(steps):
        robot_fwd.act("forward")

    # Random run
    robot_rand = Robot(grid, x, y, heading=heading)
    for _ in range(steps):
        robot_rand.act(random.choice(ACTIONS))

    print(f"\n{BOLD}Comparison over {steps} steps in unseen maze:{RESET}\n")
    sdm_rate = robot_eam.collisions / steps * 100
    fwd_rate = robot_fwd.collisions / steps * 100
    rand_rate = robot_rand.collisions / steps * 100
    print(f"  {GREEN}SDM (danger memory):{RESET}  {robot_eam.collisions} collisions ({sdm_rate:.1f}%)")
    print(f"  {RED}Always forward:{RESET}      {robot_fwd.collisions} collisions ({fwd_rate:.1f}%)")
    print(f"  {YELLOW}Random:{RESET}             {robot_rand.collisions} collisions ({rand_rate:.1f}%)")
    print()


def run(brain: Brain):
    """Main interactive loop."""
    print(f"\n{BOLD}{MAGENTA}Navigator{RESET} — danger memory")
    print(f"{DIM}The robot moves forward by default. When it crashes, it remembers.")
    print(f"Next time it senses danger, it flinches away. One memory. One instinct.{RESET}")
    print(f"{DIM}Commands: go [steps], transfer [steps], compare, stats, quit{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}nav>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}goodbye{RESET}")
            break

        if not user_input:
            continue

        if user_input == "quit":
            print(f"{DIM}goodbye{RESET}")
            break

        if user_input.startswith("go"):
            parts = user_input.split()
            steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 300
            try:
                cmd_live(brain, steps=steps)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input.startswith("transfer"):
            parts = user_input.split()
            steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
            if brain.memories == 0:
                print(f"{DIM}  no danger memories yet — use 'go' first{RESET}\n")
                continue
            try:
                cmd_transfer(brain, steps=steps)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "compare":
            if brain.memories == 0:
                print(f"{DIM}  no danger memories yet — use 'go' first{RESET}\n")
                continue
            try:
                cmd_compare(brain)
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        if user_input == "stats":
            try:
                stats = brain.heather.stats()
                print(f"\n{DIM}  Danger memories: {brain.memories}")
                print(f"  HeatherDB: {stats.get('num_locations', '?')} locations, "
                      f"{stats.get('total_writes', '?')} total writes{RESET}\n")
            except Exception as e:
                print(f"{RED}  error: {e}{RESET}\n")
            continue

        print(f"{DIM}  unknown command. try: go [steps], transfer [steps], compare, stats, quit{RESET}\n")
