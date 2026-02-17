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
STATE_SIZE = 32  # number of features in the encoded state vector
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 8
MIN_NEURONS_PER_LAYER = 8
MAX_NEURONS_PER_LAYER = 2048
DEFAULT_HIDDEN_LAYERS = (128, 128, 64)

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
    episodes: int = 10000
    max_steps: int = 2000          # was 250 — must be large for late-game navigation
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
    stall_limit_factor: int = 60
    lr_start: float = 1e-3
    lr_end: float = 2e-4
    gamma_start: float = 0.96
    gamma_end: float = 0.99
    epsilon_decay_rate: float = 8.0
    batch_size_start: int = 64
    batch_size_end: int = 128
    # --- reward shaping ---
    food_reward_base: float = 1.0      # reduced from 1.5: less greedy
    food_reward_min: float = 0.6       # floor as training progresses
    survival_reward_start: float = 0.04
    survival_reward_end: float = 0.15
    death_reward: float = -2.0
    # flood fill positioning bonus — reward agent for keeping more open space
    flood_fill_bonus_weight: float = 0.03
    # tail proximity bonus — reward agent for keeping escape route via tail
    tail_bonus_weight: float = 0.01
    # length penalty on death — dying longer is punished more
    length_death_penalty_scale: float = 0.05
    terminal_bonus_alpha_start: float = 0.05
    terminal_bonus_alpha_end: float = 0.20
    terminal_bonus_power: float = 1.5   # more superlinear: winning big matters a lot
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


# ---------------------------------------------------------------------------
# State encoding — 32-feature spatial MLP input
# ---------------------------------------------------------------------------

_ESCAPE_CACHE: dict[tuple[int, int, frozenset[tuple[int, int]]], int] = {}


def clear_escape_cache() -> None:
    """Clear flood-fill cache to free memory (call every ~100 episodes)."""
    _ESCAPE_CACHE.clear()


def _is_collision(game: SnakeGame, x: int, y: int) -> bool:
    size = game.config.grid_size
    if x < 0 or x >= size or y < 0 or y >= size:
        return True
    return (x, y) in game.snake_set


def _full_flood_fill(game: SnakeGame, start_x: int, start_y: int) -> int:
    """
    Count all cells reachable from (start_x, start_y) via BFS, treating the
    snake body as walls but allowing the tail cell (it will vacate next tick).
    Returns 0 if the start cell itself is a wall or out of bounds.
    This gives the agent a board-wide sense of available space, not just
    the 2-step lookahead of the old depth-limited version.
    """
    size = game.config.grid_size
    if start_x < 0 or start_x >= size or start_y < 0 or start_y >= size:
        return 0

    # Tail vacates next tick — treat it as passable so agent can "chase" its tail
    tail = game.snake[-1] if game.snake else None

    if (start_x, start_y) in game.snake_set and (start_x, start_y) != tail:
        return 0

    # Cache using first 12 body segments for key (balance accuracy vs speed)
    body_key = frozenset(list(game.snake)[:12])
    cache_key = (start_x * size + start_y, size, body_key)
    if cache_key in _ESCAPE_CACHE:
        return _ESCAPE_CACHE[cache_key]

    visited: set[tuple[int, int]] = {(start_x, start_y)}
    queue: list[tuple[int, int]] = [(start_x, start_y)]
    head = 0

    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= size or ny < 0 or ny >= size:
                continue
            if (nx, ny) in visited:
                continue
            # Body blocks movement, except tail which will move away
            if (nx, ny) in game.snake_set and (nx, ny) != tail:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))

    result = len(visited)
    if len(_ESCAPE_CACHE) < 8000:
        _ESCAPE_CACHE[cache_key] = result
    return result


def _nearest_body_distance(game: SnakeGame, hx: int, hy: int, dx: int, dy: int) -> float:
    """
    Scan in direction (dx, dy) from head and return normalized distance to the
    first body segment or wall encountered. Returns 1.0 if the path is fully
    clear to the board edge. Returns 0.0 if the very next cell is blocked.
    This gives the agent a continuous danger signal instead of a binary one.
    """
    size = game.config.grid_size
    max_dist = size  # maximum possible distance
    x, y = hx + dx, hy + dy
    steps = 1
    while 0 <= x < size and 0 <= y < size:
        if (x, y) in game.snake_set:
            return 1.0 - (steps / max_dist)  # closer body = lower value
        x += dx
        y += dy
        steps += 1
    # Hit wall without hitting body: return distance to wall
    return 1.0 - ((steps - 1) / max_dist)


def nearest_apple_position(game: SnakeGame) -> tuple[int, int]:
    if not game.apples:
        return game.snake[0]
    hx, hy = game.snake[0]
    best: tuple[int, int] | None = None
    best_dist = 10 ** 9
    for ax, ay in game.apples:
        dist = abs(ax - hx) + abs(ay - hy)
        if dist < best_dist:
            best_dist = dist
            best = (ax, ay)
    return best if best is not None else (hx, hy)


def _get_collision_info(game: SnakeGame, hx: int, hy: int) -> dict[str, bool]:
    """Immediate (1-step) collision check for all 4 directions."""
    return {
        "up":    _is_collision(game, hx, hy - 1),
        "down":  _is_collision(game, hx, hy + 1),
        "left":  _is_collision(game, hx - 1, hy),
        "right": _is_collision(game, hx + 1, hy),
    }


def encode_integer_state(game: SnakeGame) -> np.ndarray:
    """Legacy 12-feature encoder kept for backward compat. Not used in training."""
    hx, hy = game.snake[0]
    direction = game.direction
    ax, ay = nearest_apple_position(game)
    collisions = _get_collision_info(game, hx, hy)
    return np.array(
        [
            int(collisions["up"]),   int(collisions["down"]),
            int(collisions["left"]), int(collisions["right"]),
            int(direction == "up"),  int(direction == "down"),
            int(direction == "left"),int(direction == "right"),
            int(ay < hy), int(ay > hy),
            int(ax < hx), int(ax > hx),
        ],
        dtype=np.float32,
    )


def encode_integer_state_v2(game: SnakeGame) -> np.ndarray:
    """
    Advanced 32-feature spatial state.

    Layout:
      [0-3]   Immediate danger per direction (binary)
      [4-7]   Current direction one-hot
      [8-11]  Food direction one-hot (is food in this direction?)
      [12-13] Food offset: normalized signed dx, dy to nearest apple
      [14]    Food manhattan distance normalized by board diagonal
      [15-18] Full flood-fill fraction per adjacent direction
              (reachable cells / total board cells — full BFS, not depth-limited)
      [19]    Tail reachable from head (boolean): can the agent escape via tail?
      [20-21] Tail offset: normalized signed dx, dy to tail tip
      [22]    Tail manhattan distance normalized
      [23-26] Nearest body distance per direction (continuous 0-1, not binary)
      [27]    Snake length normalized by board capacity
      [28-31] Body density per board quadrant (NW, NE, SW, SE)
    """
    size = game.config.grid_size
    board_capacity = float(size * size)
    board_diagonal = float(size * 2 - 2)  # max manhattan distance

    hx, hy = game.snake[0]
    direction = game.direction
    ax, ay = nearest_apple_position(game)
    collisions = _get_collision_info(game, hx, hy)

    # [0-3] Immediate danger
    danger = [
        float(collisions["up"]),
        float(collisions["down"]),
        float(collisions["left"]),
        float(collisions["right"]),
    ]

    # [4-7] Direction one-hot
    dir_vec = [
        float(direction == "up"),
        float(direction == "down"),
        float(direction == "left"),
        float(direction == "right"),
    ]

    # [8-11] Food direction (binary: is food generally in this direction?)
    food_dir = [
        float(ay < hy),   # food is above
        float(ay > hy),   # food is below
        float(ax < hx),   # food is left
        float(ax > hx),   # food is right
    ]

    # [12-13] Food offset normalized (-1 to +1)
    food_dx = float(ax - hx) / float(size)
    food_dy = float(ay - hy) / float(size)

    # [14] Food distance normalized
    food_dist = float(abs(ax - hx) + abs(ay - hy)) / board_diagonal

    # [15-18] Full flood fill per direction
    fill_up    = _full_flood_fill(game, hx, hy - 1) / board_capacity
    fill_down  = _full_flood_fill(game, hx, hy + 1) / board_capacity
    fill_left  = _full_flood_fill(game, hx - 1, hy) / board_capacity
    fill_right = _full_flood_fill(game, hx + 1, hy) / board_capacity

    # [19] Tail reachable from head
    # The tail tip is the "exit" when cornered — knowing it's reachable is critical
    tx, ty = game.snake[-1]
    tail_reachable = float(_full_flood_fill(game, tx, ty) > 0)

    # [20-21] Tail offset normalized
    tail_dx = float(tx - hx) / float(size)
    tail_dy = float(ty - hy) / float(size)

    # [22] Tail distance normalized
    tail_dist = float(abs(tx - hx) + abs(ty - hy)) / board_diagonal

    # [23-26] Nearest body distance in each direction (continuous)
    body_up    = _nearest_body_distance(game, hx, hy, 0, -1)
    body_down  = _nearest_body_distance(game, hx, hy, 0,  1)
    body_left  = _nearest_body_distance(game, hx, hy, -1, 0)
    body_right = _nearest_body_distance(game, hx, hy,  1, 0)

    # [27] Snake length normalized
    snake_length_norm = float(len(game.snake)) / board_capacity

    # [28-31] Body density in each board quadrant (NW, NE, SW, SE)
    mid = size // 2
    body_list = list(game.snake)
    total_body = float(max(1, len(body_list)))
    nw = sum(1 for (bx, by) in body_list if bx < mid and by < mid) / total_body
    ne = sum(1 for (bx, by) in body_list if bx >= mid and by < mid) / total_body
    sw = sum(1 for (bx, by) in body_list if bx < mid and by >= mid) / total_body
    se = sum(1 for (bx, by) in body_list if bx >= mid and by >= mid) / total_body

    state = np.array(
        [
            # [0-3] immediate danger
            danger[0], danger[1], danger[2], danger[3],
            # [4-7] direction
            dir_vec[0], dir_vec[1], dir_vec[2], dir_vec[3],
            # [8-11] food direction
            food_dir[0], food_dir[1], food_dir[2], food_dir[3],
            # [12-14] food position
            food_dx, food_dy, food_dist,
            # [15-18] flood fill
            fill_up, fill_down, fill_left, fill_right,
            # [19-22] tail info
            tail_reachable, tail_dx, tail_dy, tail_dist,
            # [23-26] body distance
            body_up, body_down, body_left, body_right,
            # [27] length
            snake_length_norm,
            # [28-31] quadrant density
            nw, ne, sw, se,
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

    # Gamma: ramps from 0.96 -> 0.99 over training.
    # Higher gamma later means agent thinks further ahead.
    gamma = cfg.gamma_start + (cfg.gamma_end - cfg.gamma_start) * (progress**0.5)

    # Learning rate: decays from lr_start to 40% of lr_start
    lr = cfg.lr_start * (1.0 - 0.6 * progress)

    # Epsilon: fast decay in first half, plateau in second half.
    # Keeps some exploration during late-game where the agent needs to
    # discover non-greedy strategies it would never try greedily.
    if progress < 0.5:
        epsilon = max(cfg.epsilon_min, cfg.epsilon_start * math.exp(-cfg.epsilon_decay_rate * progress))
    else:
        mid_epsilon = cfg.epsilon_start * math.exp(-cfg.epsilon_decay_rate * 0.5)
        # Slow continued decay in second half rather than rushing to minimum
        remaining = mid_epsilon - cfg.epsilon_min
        epsilon = max(cfg.epsilon_min, mid_epsilon - remaining * (progress - 0.5))

    # Batch size: grows quadratically (small batches early, big batches late)
    batch_size = int(round(cfg.batch_size_start + (cfg.batch_size_end - cfg.batch_size_start) * (progress**2)))
    batch_size = max(cfg.batch_size_start, min(cfg.batch_size_end, batch_size))

    # Food reward: decays faster than before. In late training the agent should
    # stop being greedy about every apple and think about survivability first.
    # Falls from food_reward_base to food_reward_min over training.
    food_reward = max(
        cfg.food_reward_min,
        cfg.food_reward_base * (1.0 - 0.7 * progress),   # was 0.5 * progress
    )

    # Survival reward: grows with concave curve — reaches its ceiling faster
    # so mid-game training already heavily rewards staying alive.
    survival_reward = cfg.survival_reward_start + (
        cfg.survival_reward_end - cfg.survival_reward_start
    ) * (progress**0.5)   # was 0.7, now 0.5 — faster ramp

    # Terminal bonus alpha: very aggressive superlinear growth.
    # The agent should learn that a high-length terminal state is
    # exponentially more valuable than a medium one.
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

        # --- Reward shaping ---
        # Philosophy: less food greed, more survival, spatial positioning bonuses.

        # Food: agent gets credit for eating, but reward decays with training
        # to reduce the greedy apple-chasing that causes self-trapping.
        food_reward = dynamics.food_reward if new_length > old_length else 0.0

        # Survival: small per-step reward for staying alive.
        survival_reward = dynamics.survival_reward if alive else 0.0

        # Death/stall: flat penalty plus a length-scaled component.
        # Dying at length 50 should hurt far more than dying at length 5
        # because the agent has wasted far more accumulated progress.
        death_reward = 0.0
        if not alive or stalled:
            length_penalty = cfg.length_death_penalty_scale * float(len(game.snake))
            death_reward = dynamics.death_reward - length_penalty

        # Flood fill bonus: reward moves that preserve more open space.
        # Computed as the flood fill of the new head position normalized by
        # board capacity. High open space = good position. This teaches the
        # agent to prefer moves that don't box itself in, even when a greedy
        # apple move would close off space.
        flood_bonus = 0.0
        if alive and not done:
            hx, hy = game.snake[0]
            open_cells = _full_flood_fill(game, hx, hy)
            flood_fraction = open_cells / float(max(1, board_capacity))
            flood_bonus = cfg.flood_fill_bonus_weight * flood_fraction

        # Tail proximity bonus: reward keeping the tail reachable.
        # The tail is the agent's escape hatch — if it can always reach its
        # own tail via flood fill, it can never be permanently cornered.
        # This teaches the "tail-chasing" strategy that strong Snake agents use.
        tail_bonus = 0.0
        if alive and not done and len(game.snake) > 1:
            tx, ty = game.snake[-1]
            if _full_flood_fill(game, tx, ty) > 0:
                tail_bonus = cfg.tail_bonus_weight

        immediate_reward = food_reward + survival_reward + death_reward + flood_bonus + tail_bonus

        # Clip for TD stability. Terminal bonus is added after clip so a
        # high-score terminal state isn't artificially capped.
        clipped_immediate_reward = float(np.clip(immediate_reward, -3.0, 2.0))

        terminal_bonus = 0.0
        if done:
            final_score = float(game.score)
            terminal_bonus = dynamics.terminal_bonus_alpha * (
                max(0.0, final_score) ** cfg.terminal_bonus_power
            )
        reward = clipped_immediate_reward + terminal_bonus

        food_reward_total += food_reward
        survival_reward_total += survival_reward
        death_reward_total += death_reward
        safety_bonus_total += flood_bonus + tail_bonus   # reuse safety_bonus slot
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
