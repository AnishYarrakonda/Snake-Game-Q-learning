# Core Snake game state and rules, independent from GUI/training code.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random


# Bounds used by the GUI when validating user input.
MIN_GRID_SIZE = 8
MAX_GRID_SIZE = 60
MIN_CELL_SIZE = 12
MAX_CELL_SIZE = 48
MIN_SPEED_MS = 40
MAX_SPEED_MS = 500
MIN_APPLES = 1
MAX_APPLES = 100
MIN_INITIAL_LENGTH = 2


@dataclass
class SnakeConfig:
    """Runtime settings shared between the logic layer and GUI."""
    grid_size: int = 20
    cell_size: int = 28
    speed_ms: int = 100
    apples: int = 3
    initial_length: int = 3
    wrap_walls: bool = False
    show_grid: bool = True


class SnakeGame:
    """Pure game state + rules (no Tkinter/UI code)."""
    def __init__(self, config: SnakeConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Initialize a fresh board with centered snake and apples."""
        size = self.config.grid_size
        self.snake: deque[tuple[int, int]] = deque()        # ordered body, head at index 0
        self.snake_set: set[tuple[int, int]] = set()        # O(1) body collision lookup
        self.apples: set[tuple[int, int]] = set()           # apple positions
        self.free_tiles = {(x, y) for x in range(size) for y in range(size)}
        self.direction = "right"
        self.pending_direction = "right"                    # queued from input; applied next tick
        self.running = False
        self.alive = True
        self.score = 0

        self.spawn_snake(self.config.initial_length)
        self.replenish_apples()

    def spawn_snake(self, length: int) -> None:
        """Place snake so the head starts at the board center."""
        size = self.config.grid_size
        center_y = size // 2
        center_x = size // 2

        # Build body extending left from head.
        positions = [(center_x - i, center_y) for i in range(length)]
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)

        # Fallback for tiny boards / large initial length.
        if min_x < 0 or max_x >= size:
            tail_x = (size - length) // 2
            head_x = tail_x + length - 1
            positions = [(head_x - i, center_y) for i in range(length)]

        for x, y in positions:
            self.snake.append((x, y))
            self.snake_set.add((x, y))
            self.free_tiles.discard((x, y))

    def _next_head(self, direction: str) -> tuple[int, int]:
        """Translate current head by one tile in the given direction."""
        head_x, head_y = self.snake[0]
        if direction == "up":
            return head_x, head_y - 1
        if direction == "down":
            return head_x, head_y + 1
        if direction == "left":
            return head_x - 1, head_y
        return head_x + 1, head_y

    def _in_bounds(self, x: int, y: int) -> bool:
        size = self.config.grid_size
        return 0 <= x < size and 0 <= y < size

    def move(self) -> bool:
        """Advance one step. Returns False if the snake dies this tick."""
        if not self.alive:
            return False

        # Apply the latest valid input once per tick.
        self.direction = self.pending_direction
        new_x, new_y = self._next_head(self.direction)

        # Optional wrap mode: crossing an edge teleports to opposite edge.
        if self.config.wrap_walls:
            size = self.config.grid_size
            new_x %= size
            new_y %= size
        elif not self._in_bounds(new_x, new_y):
            self.alive = False
            return False

        new_head = (new_x, new_y)
        growing = new_head in self.apples
        tail = self.snake[-1]

        # Moving into current tail is allowed only if not growing
        # (because tail moves away in the same tick).
        if new_head in self.snake_set and not (not growing and new_head == tail):
            self.alive = False
            return False

        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        self.free_tiles.discard(new_head)

        if growing:
            self.apples.discard(new_head)
            self.score += 1
        else:
            old_tail = self.snake.pop()
            self.snake_set.discard(old_tail)
            self.free_tiles.add(old_tail)

        self.replenish_apples()
        return True

    def replenish_apples(self) -> None:
        """Keep spawning apples until configured count is reached or board is full."""
        target = min(self.config.apples, len(self.free_tiles))
        while len(self.apples) < target and self.free_tiles:
            # Python 3.13 random.sample requires a sequence (set is not accepted).
            pos = random.choice(tuple(self.free_tiles))
            self.apples.add(pos)
            self.free_tiles.discard(pos)

    def queue_direction(self, new_direction: str) -> None:
        """Queue an input direction; reject instant 180-degree turns."""
        opposites = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }
        if new_direction not in opposites:
            return
        if len(self.snake) > 1 and opposites[new_direction] == self.direction:
            return
        self.pending_direction = new_direction
