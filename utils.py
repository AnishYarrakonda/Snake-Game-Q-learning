# Shared training helpers: config, state encoding, model paths, and episode simulation.
from __future__ import annotations

from dataclasses import dataclass
import math
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
STATE_ENCODING_INTEGER = "integer12"
SUPPORTED_STATE_ENCODINGS = (STATE_ENCODING_INTEGER, "compact11")
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 8
MIN_NEURONS_PER_LAYER = 8
MAX_NEURONS_PER_LAYER = 2048
DEFAULT_HIDDEN_LAYERS = (128, 128, 64)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

_ESCAPE_CACHE: dict[tuple[int, int, int, frozenset[tuple[int, int]]], int] = {}


def _parse_positive_int(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value


def normalize_hidden_layers(raw_layers: object) -> tuple[int, ...]:
    if isinstance(raw_layers, int):
        layers = (raw_layers,)
    elif isinstance(raw_layers, tuple):
        layers = raw_layers
    elif isinstance(raw_layers, list):
        layers = tuple(raw_layers)
    else:
        raise ValueError("Hidden layers must be an int, list[int], or tuple[int, ...].")

    if not (MIN_HIDDEN_LAYERS <= len(layers) <= MAX_HIDDEN_LAYERS):
        raise ValueError(f"Hidden layer count must be between {MIN_HIDDEN_LAYERS} and {MAX_HIDDEN_LAYERS}.")

    validated: list[int] = []
    for idx, width in enumerate(layers, start=1):
        if not isinstance(width, int):
            raise ValueError(f"Hidden layer {idx} width must be an integer.")
        if not (MIN_NEURONS_PER_LAYER <= width <= MAX_NEURONS_PER_LAYER):
            raise ValueError(
                f"Hidden layer {idx} width must be between {MIN_NEURONS_PER_LAYER} and {MAX_NEURONS_PER_LAYER}."
            )
        validated.append(width)
    return tuple(validated)


def parse_hidden_layer_widths(layer_count: int, neurons_raw: str) -> tuple[int, ...]:
    if not (MIN_HIDDEN_LAYERS <= layer_count <= MAX_HIDDEN_LAYERS):
        raise ValueError(f"Hidden layers must be between {MIN_HIDDEN_LAYERS} and {MAX_HIDDEN_LAYERS}.")

    raw = neurons_raw.strip()
    if not raw:
        raise ValueError("Neurons per layer is required.")

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Neurons per layer is required.")

    if len(parts) == 1:
        width = _parse_positive_int(parts[0], "Neurons per layer")
        return normalize_hidden_layers(tuple(width for _ in range(layer_count)))

    if len(parts) != layer_count:
        raise ValueError(
            f"Neurons per layer must be a single integer or exactly {layer_count} comma-separated integers."
        )

    widths = tuple(_parse_positive_int(part, f"Neurons for hidden layer {idx + 1}") for idx, part in enumerate(parts))
    return normalize_hidden_layers(widths)


@dataclass
class TrainConfig:
    board_size: int = 10
    apples: int = 5
    episodes: int = 10000
    max_steps: int = 2000
    gamma: float = 0.96
    lr: float = 0.001
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.994
    batch_size: int = 128
    memory_size: int = 50_000
    replay_warmup: int = 0
    hidden_layers: tuple[int, ...] = (256, 256, 128)
    target_update_every: int = 1
    soft_tau: float = 0.01
    n_step: int = 5
    step_delay: float = 0.0
    state_encoding: str = STATE_ENCODING_INTEGER
    stall_limit_factor: int = 75
    lr_start: float = 1e-3
    lr_end: float = 2e-4
    gamma_start: float = 0.96
    gamma_end: float = 0.99
    epsilon_decay_rate: float = 8.0
    batch_size_start: int = 64
    batch_size_end: int = 128
    food_reward_base: float = 1.5
    food_reward_min: float = 0.8
    survival_reward_start: float = 0.05
    survival_reward_end: float = 0.20
    death_reward: float = -2.0
    terminal_bonus_alpha_start: float = 0.05
    terminal_bonus_alpha_end: float = 0.15
    terminal_bonus_power: float = 1.3
    q_explosion_threshold: float = 100.0

    def __post_init__(self) -> None:
        self.hidden_layers = normalize_hidden_layers(self.hidden_layers)
        if self.state_encoding == "compact11":
            self.state_encoding = STATE_ENCODING_INTEGER
        if self.state_encoding not in SUPPORTED_STATE_ENCODINGS:
            raise ValueError(f"Unsupported state encoding: {self.state_encoding}")
        if self.episodes <= 0:
            raise ValueError("episodes must be > 0.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be > 0.")
        if self.replay_warmup < 0:
            raise ValueError("replay_warmup must be >= 0.")
        if self.n_step <= 0:
            raise ValueError("n_step must be > 0.")


@dataclass(frozen=True)
class EpisodeDynamics:
    progress: float
    gamma: float
    n_step: int
    lr: float
    epsilon: float
    batch_size: int
    food_reward: float
    survival_reward: float
    death_reward: float
    terminal_bonus_alpha: float


class AgentLike(Protocol):
    epsilon: float

    def valid_action_indices(self, current_direction: str) -> list[int]: ...

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int: ...

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_direction: str,
        done: bool,
        discount: float,
    ) -> None: ...

    def train_step(self) -> dict[str, float] | None: ...
    def apply_episode_dynamics(self, dynamics: EpisodeDynamics) -> None: ...


def default_model_path(board_size: int, state_encoding: str = STATE_ENCODING_INTEGER) -> str:
    mode = "integer12" if state_encoding in SUPPORTED_STATE_ENCODINGS else state_encoding
    return os.path.join(MODELS_DIR, f"snake_dqn_{mode}_{board_size}x{board_size}.pt")


def _is_collision(game: SnakeGame, x: int, y: int) -> bool:
    size = game.config.grid_size
    if x < 0 or x >= size or y < 0 or y >= size:
        return True
    return (x, y) in game.snake_set


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


def _get_collision_info(game: SnakeGame, hx: int, hy: int) -> dict[str, bool]:
    """Get immediate collision status for all 4 directions."""
    return {
        "up": _is_collision(game, hx, hy - 1),
        "down": _is_collision(game, hx, hy + 1),
        "left": _is_collision(game, hx - 1, hy),
        "right": _is_collision(game, hx + 1, hy),
    }


def encode_integer_state(game: SnakeGame) -> np.ndarray:
    """
    Integer feature state:
    [danger_up, danger_down, danger_left, danger_right,
     dir_up, dir_down, dir_left, dir_right,
     food_up, food_down, food_left, food_right]
    """
    hx, hy = game.snake[0]
    direction = game.direction
    ax, ay = nearest_apple_position(game)
    collisions = _get_collision_info(game, hx, hy)

    state = np.array(
        [
            int(collisions["up"]),
            int(collisions["down"]),
            int(collisions["left"]),
            int(collisions["right"]),
            int(direction == "up"),
            int(direction == "down"),
            int(direction == "left"),
            int(direction == "right"),
            int(ay < hy),
            int(ay > hy),
            int(ax < hx),
            int(ax > hx),
        ],
        dtype=np.float32,
    )
    return state


def _count_escape_routes(game: SnakeGame, x: int, y: int, depth: int = 2) -> int:
    """
    Fast iterative BFS to count reachable cells within depth steps.
    Caches results using snake body snapshot for efficiency.
    """
    size = game.config.grid_size

    if x < 0 or x >= size or y < 0 or y >= size:
        return 0
    if (x, y) in game.snake_set and (x, y) != game.snake[0]:
        return 0

    body_key = frozenset(list(game.snake)[:10])
    cache_key = (x, y, depth, body_key)
    if cache_key in _ESCAPE_CACHE:
        return _ESCAPE_CACHE[cache_key]

    visited = {(x, y)}
    queue: list[tuple[int, int, int]] = [(x, y, 0)]
    head = 0

    while head < len(queue):
        cx, cy, d = queue[head]
        head += 1

        if d >= depth:
            continue

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= size or ny < 0 or ny >= size:
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in game.snake_set and (nx, ny) != game.snake[0]:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, d + 1))

    result = len(visited)
    if len(_ESCAPE_CACHE) < 5000:
        _ESCAPE_CACHE[cache_key] = result
    return result


def clear_escape_cache() -> None:
    """Clear escape route cache to free memory."""
    _ESCAPE_CACHE.clear()


def encode_integer_state_v2(game: SnakeGame) -> np.ndarray:
    """
    Enhanced 16-feature state:
    [0-3] danger per direction,
    [4-7] current direction one-hot,
    [8-11] food direction one-hot,
    [12-15] normalized safety scores per direction.
    """
    hx, hy = game.snake[0]
    direction = game.direction
    ax, ay = nearest_apple_position(game)
    collisions = _get_collision_info(game, hx, hy)

    safe_up = _count_escape_routes(game, hx, hy - 1, depth=2) / 4.0
    safe_down = _count_escape_routes(game, hx, hy + 1, depth=2) / 4.0
    safe_left = _count_escape_routes(game, hx - 1, hy, depth=2) / 4.0
    safe_right = _count_escape_routes(game, hx + 1, hy, depth=2) / 4.0

    state = np.array(
        [
            int(collisions["up"]),
            int(collisions["down"]),
            int(collisions["left"]),
            int(collisions["right"]),
            int(direction == "up"),
            int(direction == "down"),
            int(direction == "left"),
            int(direction == "right"),
            int(ay < hy),
            int(ay > hy),
            int(ax < hx),
            int(ax > hx),
            safe_up,
            safe_down,
            safe_left,
            safe_right,
        ],
        dtype=np.float32,
    )
    return state


def encode_state(game: SnakeGame, cfg: TrainConfig) -> np.ndarray:
    if cfg.state_encoding not in SUPPORTED_STATE_ENCODINGS:
        raise ValueError(f"Unsupported state encoding: {cfg.state_encoding}")
    return encode_integer_state_v2(game)


def nearest_apple_distance(game: SnakeGame) -> int:
    ax, ay = nearest_apple_position(game)
    hx, hy = game.snake[0]
    return abs(ax - hx) + abs(ay - hy)


def episode_progress(episode: int, total_episodes: int) -> float:
    if total_episodes <= 0:
        return 1.0
    return max(0.0, min(1.0, episode / float(total_episodes)))


def episode_dynamics(episode: int, cfg: TrainConfig) -> EpisodeDynamics:
    progress = episode_progress(episode, cfg.episodes)

    # Smooth schedules only: no abrupt switches in gamma/lr/epsilon/weights.
    gamma = cfg.gamma_start + (cfg.gamma_end - cfg.gamma_start) * (progress**0.5)
    lr = cfg.lr_start * (1.0 - 0.6 * progress)
    if progress < 0.5:
        epsilon = max(cfg.epsilon_min, cfg.epsilon_start * math.exp(-cfg.epsilon_decay_rate * progress))
    else:
        mid_epsilon = cfg.epsilon_start * math.exp(-cfg.epsilon_decay_rate * 0.5)
        remaining_decay = (mid_epsilon - cfg.epsilon_min) * (1.0 - (progress - 0.5) * 2.0)
        epsilon = max(cfg.epsilon_min, mid_epsilon - remaining_decay * 0.5)
    batch_size = int(round(cfg.batch_size_start + (cfg.batch_size_end - cfg.batch_size_start) * (progress**2)))
    batch_size = max(cfg.batch_size_start, min(cfg.batch_size_end, batch_size))

    food_reward = max(cfg.food_reward_min, cfg.food_reward_base * (1.0 - 0.5 * progress))
    survival_reward = cfg.survival_reward_start + (cfg.survival_reward_end - cfg.survival_reward_start) * (progress**0.7)
    terminal_bonus_alpha = cfg.terminal_bonus_alpha_start + (
        cfg.terminal_bonus_alpha_end - cfg.terminal_bonus_alpha_start
    ) * (progress**1.5)

    return EpisodeDynamics(
        progress=progress,
        gamma=gamma,
        n_step=cfg.n_step,
        lr=lr,
        epsilon=epsilon,
        batch_size=batch_size,
        food_reward=food_reward,
        survival_reward=survival_reward,
        death_reward=cfg.death_reward,
        terminal_bonus_alpha=terminal_bonus_alpha,
    )


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
    episode_index: int = 1,
    train: bool = True,
    render_step: Callable[[SnakeGame, int, int, float], None] | None = None,
    stop_flag: threading.Event | None = None,
    game: SnakeGame | None = None,
) -> tuple[int, float, int, dict[str, float]]:
    """Run one episode and optionally train the agent online from each transition."""
    if game is None:
        game = make_game(cfg)
    elif game.config.grid_size != cfg.board_size or game.config.apples != cfg.apples:
        game = make_game(cfg)
    else:
        game.reset()

    dynamics = episode_dynamics(episode_index, cfg)
    if train:
        agent.apply_episode_dynamics(dynamics)

    total_reward = 0.0
    steps_taken = 0
    board_capacity = cfg.board_size * cfg.board_size
    state = encode_state(game, cfg)
    stagnation_steps = 0
    n_step_buffer: list[tuple[np.ndarray, int, float, np.ndarray, str, bool]] = []

    food_reward_total = 0.0
    survival_reward_total = 0.0
    death_reward_total = 0.0
    safety_bonus_total = 0.0
    terminal_bonus_total = 0.0
    train_metrics: list[dict[str, float]] = []

    def flush_one_transition() -> None:
        if not n_step_buffer:
            return
        horizon = min(dynamics.n_step, len(n_step_buffer))
        reward_sum = 0.0
        for idx in range(horizon):
            reward_sum += (dynamics.gamma**idx) * n_step_buffer[idx][2]
            if n_step_buffer[idx][5]:
                horizon = idx + 1
                break

        first_state, first_action = n_step_buffer[0][0], n_step_buffer[0][1]
        next_state = n_step_buffer[horizon - 1][3]
        next_direction = n_step_buffer[horizon - 1][4]
        done = n_step_buffer[horizon - 1][5]
        discount = dynamics.gamma**horizon
        agent.remember(first_state, first_action, reward_sum, next_state, next_direction, done, discount)
        n_step_buffer.pop(0)

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

        game.queue_direction(action) # type: ignore
        alive = game.move()

        new_length = len(game.snake)
        won = bool(getattr(game, "won", False)) or (new_length >= board_capacity)
        if new_length > old_length:
            stagnation_steps = 0
        else:
            stagnation_steps += 1
        stalled = stagnation_steps > cfg.stall_limit_factor * max(1, len(game.snake))
        done = (not alive) or won or stalled
        next_state = encode_state(game, cfg)
        next_direction = game.direction

        # Dense rewards transition smoothly over training progress:
        # early learning favors food discovery, late learning favors survival.
        food_reward = dynamics.food_reward if new_length > old_length else 0.0
        survival_reward = dynamics.survival_reward if alive else 0.0
        death_reward = dynamics.death_reward if (not alive or stalled) else 0.0
        safety_bonus = 0.0
        if alive and not done:
            hx, hy = game.snake[0]
            escape_routes = _count_escape_routes(game, hx, hy, depth=2)
            safety_weight = 0.02 * (1.0 + 0.5 * dynamics.progress)
            safety_bonus = safety_weight * (escape_routes / 12.0)

        immediate_reward = food_reward + survival_reward + death_reward + safety_bonus
        # Clip immediate reward for TD stability; terminal score bonus stays separate from this clip.
        clipped_immediate_reward = float(np.clip(immediate_reward, -2.0, 2.0))

        terminal_bonus = 0.0
        if done:
            # Terminal performance bonus scales superlinearly with apples eaten.
            final_score = float(game.score)
            terminal_bonus = dynamics.terminal_bonus_alpha * (max(0.0, final_score) ** cfg.terminal_bonus_power)
        reward = clipped_immediate_reward + terminal_bonus

        food_reward_total += food_reward
        survival_reward_total += survival_reward
        death_reward_total += death_reward
        safety_bonus_total += safety_bonus
        terminal_bonus_total += terminal_bonus

        if train:
            n_step_buffer.append((state, action_idx, reward, next_state, next_direction, done))
            if len(n_step_buffer) >= dynamics.n_step:
                flush_one_transition()
            if done:
                while n_step_buffer:
                    flush_one_transition()
            metrics = agent.train_step()
            if metrics is not None:
                train_metrics.append(metrics)

        total_reward += reward

        if render_step is not None:
            render_step(game, step, new_length, agent.epsilon)

        if cfg.step_delay > 0:
            time.sleep(cfg.step_delay)

        steps_taken = step + 1
        state = next_state
        if done:
            break

    if train:
        while n_step_buffer:
            flush_one_transition()
        metrics = agent.train_step()
        if metrics is not None:
            train_metrics.append(metrics)

    step_norm = float(max(1, steps_taken))
    mean_q = float(np.mean([m["mean_q"] for m in train_metrics])) if train_metrics else 0.0
    td_error_mean = float(np.mean([m["td_error_mean"] for m in train_metrics])) if train_metrics else 0.0
    max_abs_q = float(np.max([m["max_abs_q"] for m in train_metrics])) if train_metrics else 0.0
    loss_mean = float(np.mean([m["loss"] for m in train_metrics])) if train_metrics else 0.0

    episode_stats = {
        "progress": dynamics.progress,
        "gamma": dynamics.gamma,
        "lr": dynamics.lr,
        "epsilon": dynamics.epsilon,
        "batch_size": float(dynamics.batch_size),
        "episode_length": float(steps_taken),
        "avg_food_reward": food_reward_total / step_norm,
        "avg_survival_reward": survival_reward_total / step_norm,
        "avg_death_reward": death_reward_total / step_norm,
        "avg_safety_bonus": safety_bonus_total / step_norm,
        "terminal_bonus": terminal_bonus_total,
        "mean_q": mean_q,
        "td_error_mean": td_error_mean,
        "max_abs_q": max_abs_q,
        "loss": loss_mean,
    }
    return len(game.snake), total_reward, steps_taken, episode_stats
