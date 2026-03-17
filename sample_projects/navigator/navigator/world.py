"""2D grid world with a robot that has distance sensors."""

from __future__ import annotations

import numpy as np

# Directions: N, NE, E, SE, S, SW, W, NW
DIR_OFFSETS = [
    (0, -1),   # N
    (1, -1),   # NE
    (1, 0),    # E
    (1, 1),    # SE
    (0, 1),    # S
    (-1, 1),   # SW
    (-1, 0),   # W
    (-1, -1),  # NW
]

# Heading: 0=N, 1=E, 2=S, 3=W
HEADING_NAMES = ["N", "E", "S", "W"]
HEADING_ARROWS = ["\u2191", "\u2192", "\u2193", "\u2190"]  # ↑ → ↓ ←
HEADING_DX = [0, 1, 0, -1]
HEADING_DY = [-1, 0, 1, 0]

# Actions
ACTIONS = ["forward", "turn_left", "turn_right", "backward"]


MAZE_TRAIN = [
    "####################",
    "#..................#",
    "#.####.##..........#",
    "#.....#.#.####.....#",
    "#.......#......#...#",
    "#.####..#.#....#...#",
    "#.......#.#....#...#",
    "#..##...#......#...#",
    "#..##.....####.....#",
    "#..................#",
    "#...####...........#",
    "#......#.####......#",
    "#......#.......##..#",
    "#..##..#...........#",
    "#..................#",
    "####################",
]

MAZE_TEST = [
    "####################",
    "#..................#",
    "#..#####...........#",
    "#........####......#",
    "#.##.......#.......#",
    "#.##.......#..###..#",
    "#..........#.......#",
    "#...####...........#",
    "#..................#",
    "#.......###..##....#",
    "#...........###....#",
    "#.####.............#",
    "#.#..#....####.....#",
    "#.........#........#",
    "#..................#",
    "####################",
]


class Grid:
    def __init__(self, maze: list[str]):
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.walls = set()
        for y, row in enumerate(maze):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((x, y))

    def is_wall(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return True
        return (x, y) in self.walls

    def is_open(self, x: int, y: int) -> bool:
        return not self.is_wall(x, y)


class Robot:
    def __init__(self, grid: Grid, x: int, y: int, heading: int = 0):
        self.grid = grid
        self.x = x
        self.y = y
        self.heading = heading  # 0=N, 1=E, 2=S, 3=W
        self.steps = 0
        self.collisions = 0

    def sense(self, max_dist: int = 8) -> list[float]:
        """Cast rays in 8 directions, return normalized distances [0, 1]."""
        distances = []
        for dx, dy in DIR_OFFSETS:
            dist = 0
            cx, cy = self.x, self.y
            for step in range(1, max_dist + 1):
                cx += dx
                cy += dy
                if self.grid.is_wall(cx, cy):
                    break
                dist = step
            distances.append(dist / max_dist)
        return distances

    def sense_relative(self, max_dist: int = 8) -> list[float]:
        """Sense relative to heading (front, front-right, right, ..., front-left).
        This makes the sensor readings heading-invariant."""
        absolute = self.sense(max_dist)
        # Rotate so that index 0 = ahead (based on heading)
        # heading 0=N (index 0 in DIR_OFFSETS), 1=E (index 2), 2=S (index 4), 3=W (index 6)
        rotation = self.heading * 2
        rotated = absolute[rotation:] + absolute[:rotation]
        return rotated

    def act(self, action: str) -> bool:
        """Execute an action. Returns True if movement was successful."""
        self.steps += 1

        if action == "turn_left":
            self.heading = (self.heading - 1) % 4
            return True

        if action == "turn_right":
            self.heading = (self.heading + 1) % 4
            return True

        if action == "forward":
            dx, dy = HEADING_DX[self.heading], HEADING_DY[self.heading]
            nx, ny = self.x + dx, self.y + dy
            if self.grid.is_open(nx, ny):
                self.x, self.y = nx, ny
                return True
            self.collisions += 1
            return False

        if action == "backward":
            dx, dy = HEADING_DX[self.heading], HEADING_DY[self.heading]
            nx, ny = self.x - dx, self.y - dy
            if self.grid.is_open(nx, ny):
                self.x, self.y = nx, ny
                return True
            self.collisions += 1
            return False

        return False


def render(grid: Grid, robot: Robot, trail: list[tuple[int, int]] | None = None) -> str:
    """Render the grid with the robot position."""
    trail_set = set(trail) if trail else set()
    lines = []
    for y in range(grid.rows):
        row = []
        for x in range(grid.cols):
            if x == robot.x and y == robot.y:
                row.append(HEADING_ARROWS[robot.heading])
            elif (x, y) in trail_set:
                row.append("\u00b7")  # middle dot
            elif (x, y) in grid.walls:
                row.append("\u2588")  # full block
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines)


def find_open_position(grid: Grid) -> tuple[int, int]:
    """Find a random open position in the grid."""
    import random
    while True:
        x = random.randint(1, grid.cols - 2)
        y = random.randint(1, grid.rows - 2)
        if grid.is_open(x, y):
            return x, y
