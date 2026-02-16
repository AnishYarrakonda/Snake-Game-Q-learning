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
STATE_ENCODING_INTEGER = "integer12"
SUPPORTED_STATE_ENCODINGS = (STATE_ENCODING_INTEGER, "compact11")
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 8
MIN_NEURONS_PER_LAYER = 8
MAX_NEURONS_PER_LAYER = 2048
DEFAULT_HIDDEN_LAYERS = (256, 256)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


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
    episodes: int = 30000
    max_steps: int = 250
    gamma: float = 0.90
    lr: float = 0.0003
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    batch_size: int = 1000
    memory_size: int = 80_000
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS
    target_update_every: int = 1000
    step_delay: float = 0.0
    distance_reward_shaping: bool = True
    state_encoding: str = STATE_ENCODING_INTEGER
    stall_limit_factor: int = 100
    stall_penalty: float = -5.0
    reward_step: float = -0.01
    reward_apple: float = 10.0
    penalty_death: float = -20.0
    reward_win: float = 50.0
    distance_reward_delta: float = 0.25
    survival_reward_base: float = 1.0

    def __post_init__(self) -> None:
        self.hidden_layers = normalize_hidden_layers(self.hidden_layers)
        if self.state_encoding == "compact11":
            self.state_encoding = STATE_ENCODING_INTEGER
        if self.state_encoding not in SUPPORTED_STATE_ENCODINGS:
            raise ValueError(f"Unsupported state encoding: {self.state_encoding}")


@dataclass(frozen=True)
class EpisodeDynamics:
    gamma: float
    n_step: int
    long_term_weight: float
    distance_weight: float
    survival_weight: float
    lr: float
    epsilon: float
    target_update_mode: str  # hard | soft
    target_update_every: int
    target_soft_tau: float


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
        done: bool,
        discount: float,
    ) -> None: ...

    def train_step(self) -> float | None: ...
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

    state = np.array(
        [
            int(_is_collision(game, hx, hy - 1)),
            int(_is_collision(game, hx, hy + 1)),
            int(_is_collision(game, hx - 1, hy)),
            int(_is_collision(game, hx + 1, hy)),
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


def encode_state(game: SnakeGame, cfg: TrainConfig) -> np.ndarray:
    if cfg.state_encoding not in SUPPORTED_STATE_ENCODINGS:
        raise ValueError(f"Unsupported state encoding: {cfg.state_encoding}")
    return encode_integer_state(game)


def nearest_apple_distance(game: SnakeGame) -> int:
    ax, ay = nearest_apple_position(game)
    hx, hy = game.snake[0]
    return abs(ax - hx) + abs(ay - hy)


def _linear_interp(episode: int, start_ep: int, end_ep: int, start_value: float, end_value: float) -> float:
    if episode <= start_ep:
        return start_value
    if episode >= end_ep:
        return end_value
    ratio = (episode - start_ep) / float(end_ep - start_ep)
    return start_value + ratio * (end_value - start_value)


def episode_dynamics(episode: int) -> EpisodeDynamics:
    # Gamma: 0-500 => 0.90, 500-5000 => 0.97, 5000+ => 0.99
    if episode <= 500:
        gamma = 0.90
    elif episode <= 5000:
        gamma = _linear_interp(episode, 500, 5000, 0.90, 0.97)
    else:
        gamma = 0.99

    # N-step schedule.
    if episode < 2000:
        n_step = 1
    elif episode < 5000:
        n_step = 3
    else:
        n_step = 5

    # Long-term target weighting.
    if episode <= 500:
        long_term_weight = 0.5
    elif episode <= 3000:
        long_term_weight = _linear_interp(episode, 500, 3000, 0.5, 1.0)
    else:
        long_term_weight = 1.2

    # Reward shaping schedule.
    if episode < 2000:
        distance_weight = 0.2
        survival_weight = 0.05
    elif episode < 5000:
        distance_weight = 0.1
        survival_weight = 0.1
    else:
        distance_weight = 0.05
        survival_weight = 0.2

    # Learning-rate schedule.
    if episode < 5000:
        lr = 3e-4
    elif episode < 10000:
        lr = 1e-4
    else:
        lr = 5e-5

    # Epsilon schedule with periodic exploration spikes.
    if episode <= 2000:
        epsilon = _linear_interp(episode, 1, 2000, 1.0, 0.3)
    elif episode <= 10000:
        epsilon = _linear_interp(episode, 2000, 10000, 0.3, 0.05)
    else:
        epsilon = 0.05
    if episode > 0 and ((episode - 1) % 5000) < 200:
        epsilon = max(epsilon, 0.2)

    # Target net updates: hard first, then soft.
    if episode < 3000:
        target_update_mode = "hard"
        target_update_every = 1000
        target_soft_tau = 0.0
    else:
        target_update_mode = "soft"
        target_update_every = 1000
        target_soft_tau = 0.005

    return EpisodeDynamics(
        gamma=gamma,
        n_step=n_step,
        long_term_weight=long_term_weight,
        distance_weight=distance_weight,
        survival_weight=survival_weight,
        lr=lr,
        epsilon=epsilon,
        target_update_mode=target_update_mode,
        target_update_every=target_update_every,
        target_soft_tau=target_soft_tau,
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
) -> tuple[int, float, int]:
    """Run one episode and optionally train the agent online from each transition."""
    if game is None:
        game = make_game(cfg)
    elif game.config.grid_size != cfg.board_size or game.config.apples != cfg.apples:
        game = make_game(cfg)
    else:
        game.reset()

    dynamics = episode_dynamics(episode_index)
    if train:
        agent.apply_episode_dynamics(dynamics)

    total_reward = 0.0
    steps_taken = 0
    board_capacity = cfg.board_size * cfg.board_size
    use_distance_shaping = cfg.distance_reward_shaping
    state = encode_state(game, cfg)
    stagnation_steps = 0
    n_step_buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def flush_one_transition() -> None:
        if not n_step_buffer:
            return
        horizon = min(dynamics.n_step, len(n_step_buffer))
        reward_sum = 0.0
        for idx in range(horizon):
            reward_sum += (dynamics.gamma**idx) * n_step_buffer[idx][2]
            if n_step_buffer[idx][4]:
                horizon = idx + 1
                break

        first_state, first_action = n_step_buffer[0][0], n_step_buffer[0][1]
        next_state = n_step_buffer[horizon - 1][3]
        done = n_step_buffer[horizon - 1][4]
        discount = dynamics.gamma**horizon
        agent.remember(first_state, first_action, reward_sum, next_state, done, discount)
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
                reward += cfg.distance_reward_delta * dynamics.distance_weight
            elif new_distance > old_distance:
                reward -= cfg.distance_reward_delta * dynamics.distance_weight

        if alive and not won and not stalled:
            reward += cfg.survival_reward_base * dynamics.survival_weight

        done = (not alive) or won or stalled
        next_state = encode_state(game, cfg)

        if train:
            n_step_buffer.append((state, action_idx, reward, next_state, done))
            if len(n_step_buffer) >= dynamics.n_step:
                flush_one_transition()
            if done:
                while n_step_buffer:
                    flush_one_transition()
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
