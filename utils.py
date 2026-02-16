# Shared training helpers: config, state encoding, model paths, and episode simulation.
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Callable, Protocol

import numpy as np

try:
    from .game_logic import SnakeConfig, SnakeGame
except ImportError:
    from game_logic import SnakeConfig, SnakeGame


BOARD_SIZES = (10, 20, 30, 40)
APPLE_CHOICES = (1, 3, 5, 10)
ACTIONS = ("up", "down", "left", "right")
REVERSE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}
RELATIVE_ACTIONS = ("straight", "right", "left")
TURN_RIGHT = {"up": "right", "right": "down", "down": "left", "left": "up"}
TURN_LEFT = {"up": "left", "left": "down", "down": "right", "right": "up"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    board_size: int = 10
    apples: int = 5
    episodes: int = 30000
    max_steps: int = 250
    gamma: float = 0.90
    lr: float = 0.0005
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    batch_size: int = 1000
    memory_size: int = 80_000
    hidden_dim: int = 256
    target_update_every: int = 200
    step_delay: float = 0.0
    distance_reward_shaping: bool = True
    state_encoding: str = "compact11"  # compact11 | board
    stall_limit_factor: int = 100
    stall_penalty: float = -5.0
    reward_step: float = -0.01
    reward_apple: float = 10.0
    penalty_death: float = -20.0
    reward_win: float = 50.0
    distance_reward_delta: float = 0.25


class AgentLike(Protocol):
    epsilon: float

    def valid_action_indices(self, current_direction: str) -> list[int]: ...

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int: ...

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None: ...

    def train_step(self) -> float | None: ...


def default_model_path(board_size: int, state_encoding: str = "compact11") -> str:
    mode = "compact11" if state_encoding == "compact11" else "board"
    return os.path.join(MODELS_DIR, f"snake_dqn_{mode}_{board_size}x{board_size}.pt")


def encode_board_state(game: SnakeGame, board_size: int) -> np.ndarray:
    """
    Flattened board encoding:
    - 0.0: empty
    - 0.5: apple
    - -0.5: snake body
    - 1.0: snake head
    """
    board = np.zeros((board_size, board_size), dtype=np.float32)

    for x, y in game.apples:
        board[y, x] = 0.5

    for idx, (x, y) in enumerate(game.snake):
        if idx == 0:
            board[y, x] = 1.0
        else:
            board[y, x] = -0.5

    return board.reshape(-1)


def _is_collision(game: SnakeGame, x: int, y: int) -> bool:
    size = game.config.grid_size
    if x < 0 or x >= size or y < 0 or y >= size:
        return True
    return (x, y) in game.snake_set


def _next_xy(x: int, y: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return x, y - 1
    if direction == "down":
        return x, y + 1
    if direction == "left":
        return x - 1, y
    return x + 1, y


def nearest_apple_position(game: SnakeGame) -> tuple[int, int]:
    if not game.apples:
        return game.snake[0]
    hx, hy = game.snake[0]
    best = None
    best_dist = 10**9
    for ax, ay in game.apples:
        dist = abs(ax - hx) + abs(ay - hy)
        if dist < best_dist:
            best_dist = dist
            best = (ax, ay)
    return best if best is not None else (hx, hy)


def encode_compact_state(game: SnakeGame) -> np.ndarray:
    """
    Patrick Loeber style compact state:
    [danger_straight, danger_right, danger_left,
     dir_left, dir_right, dir_up, dir_down,
     food_left, food_right, food_up, food_down]
    """
    hx, hy = game.snake[0]
    direction = game.direction
    right_dir = TURN_RIGHT[direction]
    left_dir = TURN_LEFT[direction]

    sx, sy = _next_xy(hx, hy, direction)
    rx, ry = _next_xy(hx, hy, right_dir)
    lx, ly = _next_xy(hx, hy, left_dir)

    ax, ay = nearest_apple_position(game)

    state = np.array(
        [
            int(_is_collision(game, sx, sy)),
            int(_is_collision(game, rx, ry)),
            int(_is_collision(game, lx, ly)),
            int(direction == "left"),
            int(direction == "right"),
            int(direction == "up"),
            int(direction == "down"),
            int(ax < hx),
            int(ax > hx),
            int(ay < hy),
            int(ay > hy),
        ],
        dtype=np.float32,
    )
    return state


def encode_state(game: SnakeGame, cfg: TrainConfig) -> np.ndarray:
    if cfg.state_encoding == "compact11":
        return encode_compact_state(game)
    return encode_board_state(game, cfg.board_size)


def nearest_apple_distance(game: SnakeGame) -> int:
    ax, ay = nearest_apple_position(game)
    hx, hy = game.snake[0]
    return abs(ax - hx) + abs(ay - hy)


def chunked_episode_stats(
    values: list[float],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-chunk summaries for plotting.
    Returns:
    - x_end: episode index at each chunk end
    - mean
    - median
    - q1 (25th percentile)
    - q3 (75th percentile)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty, empty

    x_end: list[float] = []
    means: list[float] = []
    medians: list[float] = []
    q1s: list[float] = []
    q3s: list[float] = []

    for start in range(0, arr.size, chunk_size):
        chunk = arr[start : start + chunk_size]
        x_end.append(float(start + chunk.size))
        means.append(float(np.mean(chunk)))
        medians.append(float(np.median(chunk)))
        q1s.append(float(np.percentile(chunk, 25)))
        q3s.append(float(np.percentile(chunk, 75)))

    return (
        np.asarray(x_end, dtype=np.float32),
        np.asarray(means, dtype=np.float32),
        np.asarray(medians, dtype=np.float32),
        np.asarray(q1s, dtype=np.float32),
        np.asarray(q3s, dtype=np.float32),
    )


def chunked_median(values: list[float], chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute median value per fixed-size chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty

    x_end: list[float] = []
    medians: list[float] = []
    for start in range(0, arr.size, chunk_size):
        chunk = arr[start : start + chunk_size]
        x_end.append(float(start + chunk.size))
        medians.append(float(np.median(chunk)))

    return np.asarray(x_end, dtype=np.float32), np.asarray(medians, dtype=np.float32)


def chunked_mean(values: list[float], chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean value per fixed-size chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty

    x_end: list[float] = []
    means: list[float] = []
    for start in range(0, arr.size, chunk_size):
        chunk = arr[start : start + chunk_size]
        x_end.append(float(start + chunk.size))
        means.append(float(np.mean(chunk)))

    return np.asarray(x_end, dtype=np.float32), np.asarray(means, dtype=np.float32)


def make_game(cfg: TrainConfig) -> SnakeGame:
    game_cfg = SnakeConfig(
        grid_size=cfg.board_size,
        cell_size=18,
        speed_ms=80,
        apples=cfg.apples,
        initial_length=3,
        wrap_walls=False,
        show_grid=True,
    )
    return SnakeGame(game_cfg)


def run_episode(
    agent: AgentLike,
    cfg: TrainConfig,
    train: bool = True,
    render_step: Callable[[SnakeGame, int, int, float], None] | None = None,
    stop_flag: threading.Event | None = None,
    game: SnakeGame | None = None,
) -> tuple[int, float, int]:
    """Run one episode and optionally train the agent online from each transition."""
    if game is None:
        game = make_game(cfg)
    elif game.config.grid_size != cfg.board_size or game.config.apples != cfg.apples:
        game = make_game(cfg)
    else:
        game.reset()

    total_reward = 0.0
    steps_taken = 0
    board_capacity = cfg.board_size * cfg.board_size
    use_distance_shaping = cfg.distance_reward_shaping
    state = encode_state(game, cfg)
    stagnation_steps = 0

    for step in range(cfg.max_steps):
        if stop_flag and stop_flag.is_set():
            break

        valid_actions = agent.valid_action_indices(game.direction)
        action_idx = agent.select_action(state, valid_actions, explore=train)
        action_to_direction = getattr(agent, "action_to_direction", None)
        if callable(action_to_direction):
            action = action_to_direction(game.direction, action_idx)
        else:
            action = ACTIONS[action_idx]

        old_length = len(game.snake)
        old_distance = nearest_apple_distance(game) if use_distance_shaping else 0

        game.queue_direction(action)
        alive = game.move()

        new_length = len(game.snake)
        won = bool(getattr(game, "won", False)) or (new_length >= board_capacity)
        if new_length > old_length:
            stagnation_steps = 0
        else:
            stagnation_steps += 1
        stalled = stagnation_steps > cfg.stall_limit_factor * max(1, len(game.snake))

        reward = cfg.reward_step
        if not alive:
            reward = cfg.penalty_death
        elif won:
            reward = cfg.reward_win
        elif stalled:
            reward = cfg.stall_penalty
        elif new_length > old_length:
            reward = cfg.reward_apple
        elif use_distance_shaping:
            new_distance = nearest_apple_distance(game)
            if new_distance < old_distance:
                reward += cfg.distance_reward_delta
            elif new_distance > old_distance:
                reward -= cfg.distance_reward_delta

        done = (not alive) or won or stalled
        next_state = encode_state(game, cfg)

        if train:
            agent.remember(state, action_idx, reward, next_state, done)
            agent.train_step()

        total_reward += reward

        if render_step is not None:
            render_step(game, step, new_length, agent.epsilon)

        if cfg.step_delay > 0:
            time.sleep(cfg.step_delay)

        steps_taken = step + 1
        state = next_state
        if done:
            break

    return len(game.snake), total_reward, steps_taken
