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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class TrainConfig:
    board_size: int = 20
    apples: int = 3
    episodes: int = 3000
    max_steps: int = 1200
    gamma: float = 0.95
    lr: float = 0.0008
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    batch_size: int = 512
    memory_size: int = 100_000
    hidden_dim: int = 256
    target_update_every: int = 250
    step_delay: float = 0.0
    distance_reward_shaping: bool = True


class AgentLike(Protocol):
    epsilon: float

    def valid_action_indices(self, current_direction: str) -> list[int]: ...

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int: ...

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None: ...

    def train_step(self) -> float | None: ...


def default_model_path(board_size: int) -> str:
    return os.path.join(MODELS_DIR, f"snake_dqn_{board_size}x{board_size}.pt")


def encode_state(game: SnakeGame, board_size: int) -> np.ndarray:
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

    for x, y in list(game.snake)[1:]:
        board[y, x] = -0.5

    head_x, head_y = game.snake[0]
    board[head_y, head_x] = 1.0

    return board.reshape(-1)


def nearest_apple_distance(game: SnakeGame) -> int:
    if not game.apples:
        return 0
    hx, hy = game.snake[0]
    best = 10**9
    for ax, ay in game.apples:
        dist = abs(ax - hx) + abs(ay - hy)
        if dist < best:
            best = dist
    return best


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
) -> tuple[int, float, int]:
    """Run one episode and optionally train the agent online from each transition."""
    game = make_game(cfg)
    total_reward = 0.0

    for step in range(cfg.max_steps):
        if stop_flag and stop_flag.is_set():
            break

        state = encode_state(game, cfg.board_size)
        valid_actions = agent.valid_action_indices(game.direction)
        action_idx = agent.select_action(state, valid_actions, explore=train)
        action = ACTIONS[action_idx]

        old_length = len(game.snake)
        old_distance = nearest_apple_distance(game) if cfg.distance_reward_shaping else 0

        game.queue_direction(action)
        alive = game.move()

        new_length = len(game.snake)
        new_distance = nearest_apple_distance(game) if cfg.distance_reward_shaping else 0

        # Reward shaping gives the agent denser feedback than apple/death only.
        reward = 0.01
        if not alive:
            reward = -1.0
        elif new_length > old_length:
            reward = 1.0
        elif cfg.distance_reward_shaping:
            if new_distance < old_distance:
                reward += 0.03
            elif new_distance > old_distance:
                reward -= 0.03

        done = not alive
        next_state = encode_state(game, cfg.board_size)

        if train:
            agent.remember(state, action_idx, reward, next_state, done)
            agent.train_step()

        total_reward += reward

        if render_step is not None:
            render_step(game, step, new_length, agent.epsilon)

        if cfg.step_delay > 0:
            time.sleep(cfg.step_delay)

        if done:
            break

    return len(game.snake), total_reward, step + 1 #type: ignore
