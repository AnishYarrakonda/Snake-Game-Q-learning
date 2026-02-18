# DQN model and agent implementation for Snake.
from __future__ import annotations

from collections import deque
import random
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .utils import (
        ACTIONS,
        BOARD_STATE_CHANNELS,
        REVERSE_DIRECTION,
        EpisodeDynamics,
        STATE_ENCODING_BOARD,
        STATE_ENCODING_INTEGER,
        STATE_SIZE,
        SUPPORTED_STATE_ENCODINGS,
        TrainConfig,
        normalize_hidden_layers,
    )
except ImportError:
    from utils import (
        ACTIONS,
        BOARD_STATE_CHANNELS,
        REVERSE_DIRECTION,
        EpisodeDynamics,
        STATE_ENCODING_BOARD,
        STATE_ENCODING_INTEGER,
        STATE_SIZE,
        SUPPORTED_STATE_ENCODINGS,
        TrainConfig,
        normalize_hidden_layers,
    )

VALID_ACTIONS_BY_DIRECTION = {
    direction: [idx for idx, action in enumerate(ACTIONS) if action != REVERSE_DIRECTION[direction]]
    for direction in ACTIONS
}


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with numpy-backed priorities."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6) -> None:
        self.capacity = max(1, int(capacity))
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.storage: list[tuple[np.ndarray, int, float, np.ndarray, str, bool, float]] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0

    @property
    def maxlen(self) -> int:
        return self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def __iter__(self):
        return iter(self.storage)

    def append(self, transition: tuple[np.ndarray, int, float, np.ndarray, str, bool, float]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition

        max_prio = float(self.priorities[: max(1, len(self.storage))].max()) if len(self.storage) > 1 else 1.0
        self.priorities[self.position] = max(1e-3, max_prio)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[list[tuple[np.ndarray, int, float, np.ndarray, str, bool, float]], np.ndarray, np.ndarray]:
        size = len(self.storage)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priorities = self.priorities[:size].astype(np.float64)
        if not np.isfinite(priorities).all() or float(priorities.sum()) <= 0.0:
            priorities = np.ones(size, dtype=np.float64)
        priorities = np.clip(priorities, 1e-8, None)

        scaled = np.power(priorities, self.alpha)
        total = float(scaled.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(size, 1.0 / float(size), dtype=np.float64)
        else:
            probs = scaled / total
            probs = probs / float(np.sum(probs))

        sample_size = min(size, max(1, int(batch_size)))
        indices = np.random.choice(size, size=sample_size, replace=False, p=probs)
        batch = [self.storage[int(idx)] for idx in indices]

        weights = np.power(size * probs[indices], -float(beta)).astype(np.float32)
        max_w = float(np.max(weights))
        if max_w > 0.0:
            weights /= max_w
        else:
            weights.fill(1.0)
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        abs_errors = np.abs(td_errors).astype(np.float32)
        for idx, error in zip(indices, abs_errors):
            if 0 <= int(idx) < len(self.storage):
                self.priorities[int(idx)] = float(max(self.epsilon, error))


class CNNQNetwork(nn.Module):
    """Compact dueling CNN for board-state tensors [C, H, W]."""

    def __init__(self, input_channels: int, output_size: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        value = self.value_head(z)
        adv = self.adv_head(z)
        return value + adv - adv.mean(dim=1, keepdim=True)


class MLPQNetwork(nn.Module):
    """Dueling MLP network for compact vector state."""

    def __init__(self, input_size: int, hidden_layers: tuple[int, ...], output_size: int = 4) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty.")

        layers: list[nn.Module] = []
        in_features = input_size
        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            in_features = width
        self.trunk = nn.Sequential(*layers)

        head_width = max(64, in_features // 2)
        self.value_head = nn.Sequential(
            nn.Linear(in_features, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(in_features, head_width),
            nn.SiLU(),
            nn.Linear(head_width, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        values = self.value_head(features)
        advantages = self.adv_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class SnakeDQNAgent:
    """DQN agent with replay buffer, target network, and epsilon-greedy policy."""

    def __init__(self, cfg: TrainConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        if cfg.state_encoding not in SUPPORTED_STATE_ENCODINGS:
            raise ValueError(f"Unsupported state encoding: {cfg.state_encoding}")
        self.action_space = ACTIONS
        if device is not None:
            self.device = device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.uses_cnn = cfg.state_encoding == STATE_ENCODING_BOARD
        hidden_layers = normalize_hidden_layers(cfg.hidden_layers)
        if self.uses_cnn:
            self.policy_net = CNNQNetwork(
                input_channels=BOARD_STATE_CHANNELS,
                output_size=len(self.action_space),
            ).to(self.device)
            self.target_net = CNNQNetwork(
                input_channels=BOARD_STATE_CHANNELS,
                output_size=len(self.action_space),
            ).to(self.device)
        else:
            self.policy_net = MLPQNetwork(
                STATE_SIZE,
                hidden_layers=hidden_layers,
                output_size=len(self.action_space),
            ).to(self.device)
            self.target_net = MLPQNetwork(
                STATE_SIZE,
                hidden_layers=hidden_layers,
                output_size=len(self.action_space),
            ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        if cfg.use_prioritized_replay:
            self.memory: PrioritizedReplayBuffer | deque[tuple[np.ndarray, int, float, np.ndarray, str, bool, float]] = (
                PrioritizedReplayBuffer(cfg.memory_size, alpha=cfg.per_alpha)
            )
        else:
            self.memory = deque(maxlen=cfg.memory_size)

        self.epsilon = cfg.epsilon_start
        self.learn_steps = 0
        self.gamma = cfg.gamma
        self.batch_size_current = cfg.batch_size
        # Keep training simple: no replay warmup phase.
        self.replay_warmup = 0
        self.soft_tau = cfg.soft_tau
        self.best_score = 0.0
        self.loaded_legacy_payload = False

    def valid_action_indices(self, current_direction: str) -> list[int]:
        return VALID_ACTIONS_BY_DIRECTION[current_direction]

    def action_to_direction(self, current_direction: str, action_idx: int) -> str:
        return ACTIONS[action_idx]

    @staticmethod
    def _safe_action_indices_from_vector_state(state: np.ndarray, valid_indices: list[int]) -> list[int]:
        # Vector state uses danger flags at indices [0:4] for [up, down, left, right].
        if state.ndim != 1 or state.shape[0] < 4:
            return valid_indices
        safe = [idx for idx in valid_indices if float(state[idx]) < 0.5]
        return safe if safe else valid_indices

    @staticmethod
    def _safe_action_indices_from_board_state(state: np.ndarray, valid_indices: list[int]) -> list[int]:
        # Board state channels:
        # 0=head, 2=tail, 3=apple, 4=occupied.
        if state.ndim != 3 or state.shape[0] < 5:
            return valid_indices
        head_map = state[0]
        if float(head_map.sum()) <= 0.0:
            return valid_indices
        hy, hx = np.unravel_index(int(np.argmax(head_map)), head_map.shape)
        occupied = state[4]
        tail_map = state[2] if state.shape[0] > 2 else np.zeros_like(occupied)
        apple_map = state[3] if state.shape[0] > 3 else np.zeros_like(occupied)
        height, width = occupied.shape

        deltas = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0),   # right
        }
        safe: list[int] = []
        for idx in valid_indices:
            dx, dy = deltas.get(idx, (0, 0))
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            blocked = float(occupied[ny, nx]) > 0.5
            if blocked:
                # Moving into tail is often safe if tail will vacate and no apple sits there.
                tail_cell = float(tail_map[ny, nx]) > 0.5
                apple_cell = float(apple_map[ny, nx]) > 0.5
                if tail_cell and not apple_cell:
                    safe.append(idx)
                continue
            safe.append(idx)
        return safe if safe else valid_indices

    def _safe_action_indices_from_state(self, state: np.ndarray, valid_indices: list[int]) -> list[int]:
        if self.uses_cnn:
            return self._safe_action_indices_from_board_state(state, valid_indices)
        return self._safe_action_indices_from_vector_state(state, valid_indices)

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            safe_indices = self._safe_action_indices_from_state(state, valid_indices)
            if safe_indices and random.random() < 0.9:
                return random.choice(safe_indices)
            return random.choice(valid_indices)

        state_t = torch.from_numpy(state).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.policy_net(state_t).squeeze(0)

        # Argmax only over valid actions to avoid extra mask allocations.
        valid_q = q_values[valid_indices]
        best_in_valid = int(torch.argmax(valid_q).item())
        return int(valid_indices[best_in_valid])

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_direction: str,
        done: bool,
        discount: float,
    ) -> None:
        self.memory.append((state, action, reward, next_state, next_direction, done, discount))

    def apply_episode_dynamics(self, dynamics: EpisodeDynamics) -> None:
        self.gamma = dynamics.gamma
        self.epsilon = dynamics.epsilon
        self.batch_size_current = max(1, int(dynamics.batch_size))
        for group in self.optimizer.param_groups:
            group["lr"] = dynamics.lr

    def train_step(self) -> dict[str, float] | None:
        if len(self.memory) == 0:
            return None

        sample_size = min(len(self.memory), self.batch_size_current)
        sample_indices: np.ndarray | None = None
        sample_weights_np: np.ndarray
        if isinstance(self.memory, PrioritizedReplayBuffer):
            beta_progress = min(1.0, self.learn_steps / float(max(1, self.cfg.per_beta_anneal_steps)))
            beta = self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * beta_progress
            batch, sample_indices, sample_weights_np = self.memory.sample(sample_size, beta=beta)
        else:
            batch = random.sample(self.memory, sample_size)
            sample_weights_np = np.ones(sample_size, dtype=np.float32)

        state_shape = batch[0][0].shape
        states_np = np.empty((sample_size, *state_shape), dtype=np.float32)
        actions_np = np.empty(sample_size, dtype=np.int64)
        rewards_np = np.empty(sample_size, dtype=np.float32)
        next_states_np = np.empty((sample_size, *state_shape), dtype=np.float32)
        dones_np = np.empty(sample_size, dtype=np.float32)
        discounts_np = np.empty(sample_size, dtype=np.float32)
        next_directions: list[str] = []

        for i, (state, action, reward, next_state, next_dir, done, discount) in enumerate(batch):
            states_np[i] = state
            actions_np[i] = action
            rewards_np[i] = reward
            next_states_np[i] = next_state
            dones_np[i] = done
            discounts_np[i] = discount
            next_directions.append(next_dir)

        states_t = torch.from_numpy(states_np).to(self.device)
        actions_t = torch.from_numpy(actions_np).to(self.device)
        rewards_t = torch.from_numpy(rewards_np).to(self.device)
        next_states_t = torch.from_numpy(next_states_np).to(self.device)
        dones_t = torch.from_numpy(dones_np).to(self.device)
        discounts_t = torch.from_numpy(discounts_np).to(self.device)
        sample_weights_t = torch.from_numpy(sample_weights_np).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN with invalid-action masking on action selection.
            next_policy_q = self.policy_net(next_states_t)
            masked_next_policy_q = torch.full_like(next_policy_q, -1e9)
            for idx, direction in enumerate(next_directions):
                valid_indices = VALID_ACTIONS_BY_DIRECTION[direction]
                safe_indices = self._safe_action_indices_from_state(next_states_np[idx], valid_indices)
                allowed_indices = safe_indices if safe_indices else valid_indices
                masked_next_policy_q[idx, allowed_indices] = next_policy_q[idx, allowed_indices]
            next_actions = masked_next_policy_q.argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + discounts_t * next_q * (1.0 - dones_t)
        td_error = target_q - current_q

        abs_td = td_error.abs()
        per_sample_loss = torch.where(abs_td < 1.0, 0.5 * td_error.pow(2), abs_td - 0.5)
        loss = (per_sample_loss * sample_weights_t).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if isinstance(self.memory, PrioritizedReplayBuffer) and sample_indices is not None:
            self.memory.update_priorities(sample_indices, abs_td.detach().cpu().numpy())

        # Continuous soft target update each optimization step.
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.mul_(1.0 - self.soft_tau).add_(policy_param.data, alpha=self.soft_tau)

        self.learn_steps += 1
        with torch.no_grad():
            mean_q = float(current_q.mean().item())
            td_error_mean = float(td_error.abs().mean().item())
            max_abs_q = float(current_q.abs().max().item())

        return {
            "loss": float(loss.item()),
            "mean_q": mean_q,
            "td_error_mean": td_error_mean,
            "max_abs_q": max_abs_q,
        }

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    @staticmethod
    def _torch_load(path: str, map_location: Any) -> dict:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        if not state:
            return
        python_state = state.get("python_random")
        numpy_state = state.get("numpy_random")
        torch_cpu_state = state.get("torch_cpu")
        torch_cuda_state = state.get("torch_cuda")
        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_cpu_state is not None:
            cpu_state_tensor = torch_cpu_state
            if not isinstance(cpu_state_tensor, torch.Tensor):
                cpu_state_tensor = torch.as_tensor(cpu_state_tensor, dtype=torch.uint8)
            if cpu_state_tensor.dtype != torch.uint8:
                cpu_state_tensor = cpu_state_tensor.to(dtype=torch.uint8)
            cpu_state_tensor = cpu_state_tensor.contiguous().view(-1).cpu()
            try:
                torch.random.set_rng_state(cpu_state_tensor)
            except RuntimeError:
                # Older/newer torch builds may serialize RNG state with incompatible shape/size.
                # Skip RNG restore instead of failing model/checkpoint loading.
                pass
        if torch_cuda_state is not None and torch.cuda.is_available():
            try:
                cuda_states: list[torch.Tensor] = []
                for raw_state in torch_cuda_state:
                    state_tensor = raw_state
                    if not isinstance(state_tensor, torch.Tensor):
                        state_tensor = torch.as_tensor(state_tensor, dtype=torch.uint8)
                    if state_tensor.dtype != torch.uint8:
                        state_tensor = state_tensor.to(dtype=torch.uint8)
                    cuda_states.append(state_tensor.contiguous().view(-1).cpu())
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _new_replay_buffer(self, capacity: int | None = None):
        cap = int(capacity or self.cfg.memory_size)
        if self.cfg.use_prioritized_replay:
            return PrioritizedReplayBuffer(cap, alpha=self.cfg.per_alpha)
        return deque(maxlen=cap)

    def save_checkpoint(self, path: str, episode_index: int, replay_buffer) -> None:
        """Save full training state to allow exact resume."""
        replay_items = list(replay_buffer)
        replay_kind = "prioritized" if isinstance(replay_buffer, PrioritizedReplayBuffer) else "uniform"
        replay_priorities: list[float] | None = None
        replay_position: int | None = None
        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            replay_priorities = replay_buffer.priorities.tolist()
            replay_position = int(replay_buffer.position)
        payload = {
            "checkpoint_version": 1,
            "episode_index": int(episode_index),
            "board_size": self.cfg.board_size,
            "state_size": STATE_SIZE,
            "state_encoding": self.cfg.state_encoding,
            "action_space_size": len(self.action_space),
            "hidden_layers": list(self.cfg.hidden_layers),
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": replay_items,
            "replay_buffer_maxlen": getattr(replay_buffer, "maxlen", self.cfg.memory_size) or self.cfg.memory_size,
            "replay_kind": replay_kind,
            "replay_priorities": replay_priorities,
            "replay_position": replay_position,
            "agent_state": {
                "epsilon": float(self.epsilon),
                "gamma": float(self.gamma),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "batch_size_current": int(self.batch_size_current),
                "replay_warmup": int(self.replay_warmup),
                "soft_tau": float(self.soft_tau),
                "learn_steps": int(self.learn_steps),
                "best_score": float(self.best_score),
            },
            "rng_state": self._capture_rng_state(),
            "cfg": vars(self.cfg),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> tuple[int, object]:
        """Load full training state and return (episode_index, replay_buffer)."""
        payload = self._torch_load(path, map_location=self.device)
        # Backward compatibility: old saves only had model weights and metadata.
        if "policy_state_dict" not in payload or "optimizer_state_dict" not in payload:
            self.load(path)
            self.loaded_legacy_payload = True
            replay_buffer = self._new_replay_buffer(self.cfg.memory_size)
            self.memory = replay_buffer
            self.batch_size_current = self.cfg.batch_size
            self.replay_warmup = 0
            self.soft_tau = self.cfg.soft_tau
            return 0, replay_buffer

        if "target_state_dict" not in payload:
            raise ValueError("Checkpoint is missing target_state_dict.")

        board_size = int(payload.get("board_size", self.cfg.board_size))
        if board_size != self.cfg.board_size:
            raise ValueError(
                f"Checkpoint board size ({board_size}) does not match current board size ({self.cfg.board_size})."
            )
        payload_state_encoding = str(
            payload.get("state_encoding", payload.get("cfg", {}).get("state_encoding", STATE_ENCODING_INTEGER))
        )
        if payload_state_encoding == "compact11":
            payload_state_encoding = STATE_ENCODING_INTEGER
        if payload_state_encoding != self.cfg.state_encoding:
            raise ValueError(
                f"Checkpoint state encoding ({payload_state_encoding}) does not match current state encoding "
                f"({self.cfg.state_encoding})."
            )
        payload_action_space_size = int(payload.get("action_space_size", len(ACTIONS)))
        if payload_action_space_size != len(self.action_space):
            raise ValueError(
                f"Checkpoint action space ({payload_action_space_size}) does not match current action space "
                f"({len(self.action_space)})."
            )
        if not self.uses_cnn:
            payload_hidden_layers = self._hidden_layers_from_payload(payload, self.cfg.hidden_layers)
            if payload_hidden_layers != self.cfg.hidden_layers:
                raise ValueError(
                    f"Checkpoint hidden layers {payload_hidden_layers} do not match current hidden layers "
                    f"{self.cfg.hidden_layers}."
                )

        self.policy_net.load_state_dict(payload["policy_state_dict"])
        self.target_net.load_state_dict(payload["target_state_dict"])
        self.target_net.eval()

        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._move_optimizer_state_to_device()

        replay_items = payload.get("replay_buffer", [])
        replay_maxlen = int(payload.get("replay_buffer_maxlen", self.cfg.memory_size))
        replay_kind = str(payload.get("replay_kind", "uniform"))
        if replay_kind == "prioritized" or self.cfg.use_prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(replay_maxlen, alpha=self.cfg.per_alpha)
            for item in replay_items:
                replay_buffer.append(item)
            raw_priorities = payload.get("replay_priorities")
            if isinstance(raw_priorities, list) and raw_priorities:
                arr = np.asarray(raw_priorities, dtype=np.float32)
                replay_buffer.priorities[: min(arr.size, replay_buffer.maxlen)] = arr[: replay_buffer.maxlen]
            replay_buffer.position = int(payload.get("replay_position", replay_buffer.position))
        else:
            replay_buffer = deque(replay_items, maxlen=replay_maxlen)
        self.memory = replay_buffer

        agent_state = payload.get("agent_state", {})
        self.epsilon = float(agent_state.get("epsilon", self.cfg.epsilon_start))
        self.gamma = float(agent_state.get("gamma", self.cfg.gamma))
        restored_lr = float(agent_state.get("lr", self.cfg.lr))
        for group in self.optimizer.param_groups:
            group["lr"] = restored_lr
        self.batch_size_current = int(agent_state.get("batch_size_current", self.cfg.batch_size))
        self.replay_warmup = 0
        self.soft_tau = float(agent_state.get("soft_tau", self.cfg.soft_tau))
        self.learn_steps = int(agent_state.get("learn_steps", 0))
        self.best_score = float(agent_state.get("best_score", 0.0))

        self._restore_rng_state(payload.get("rng_state", {}))
        self.loaded_legacy_payload = False
        loaded_episode = int(payload.get("episode_index", 0))
        return loaded_episode, replay_buffer

    def save(self, path: str) -> None:
        """Save weights-only model payload (not a resumable training checkpoint)."""
        payload = {
            "board_size": self.cfg.board_size,
            "state_encoding": self.cfg.state_encoding,
            "action_space_size": len(self.action_space),
            "hidden_layers": list(self.cfg.hidden_layers),
            "state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "cfg": vars(self.cfg),
        }
        torch.save(payload, path)

    @staticmethod
    def _hidden_layers_from_payload(payload: dict, fallback: tuple[int, ...]) -> tuple[int, ...]:
        cfg_data = payload.get("cfg", {})
        if "hidden_layers" in payload:
            raw_layers = payload["hidden_layers"]
        elif "hidden_layers" in cfg_data:
            raw_layers = cfg_data["hidden_layers"]
        elif "hidden_dim" in payload:
            width = int(payload["hidden_dim"])
            raw_layers = [width, width]
        elif "hidden_dim" in cfg_data:
            width = int(cfg_data["hidden_dim"])
            raw_layers = [width, width]
        else:
            raw_layers = fallback
        return normalize_hidden_layers(raw_layers)

    def load(self, path: str) -> None:
        """Load weights-only model payload."""
        payload = self._torch_load(path, map_location=self.device)
        board_size = int(payload.get("board_size", self.cfg.board_size))
        if board_size != self.cfg.board_size:
            raise ValueError(
                f"Model board size ({board_size}) does not match current board size ({self.cfg.board_size})."
            )
        payload_state_encoding = str(
            payload.get("state_encoding", payload.get("cfg", {}).get("state_encoding", STATE_ENCODING_INTEGER))
        )
        if payload_state_encoding == "compact11":
            payload_state_encoding = STATE_ENCODING_INTEGER
        if payload_state_encoding != self.cfg.state_encoding:
            raise ValueError(
                f"Model state encoding ({payload_state_encoding}) does not match current state encoding "
                f"({self.cfg.state_encoding})."
            )
        payload_action_space_size = int(payload.get("action_space_size", len(ACTIONS)))
        if payload_action_space_size != len(self.action_space):
            raise ValueError(
                f"Model action space ({payload_action_space_size}) does not match current action space "
                f"({len(self.action_space)})."
            )
        if not self.uses_cnn:
            payload_hidden_layers = self._hidden_layers_from_payload(payload, self.cfg.hidden_layers)
            if payload_hidden_layers != self.cfg.hidden_layers:
                raise ValueError(
                    f"Model hidden layers {payload_hidden_layers} do not match current hidden layers {self.cfg.hidden_layers}."
                )

        state_dict = payload.get("state_dict", payload.get("policy_state_dict"))
        if state_dict is None:
            raise ValueError("Model payload does not contain state_dict or policy_state_dict.")
        self.policy_net.load_state_dict(state_dict)
        if "target_state_dict" in payload:
            self.target_net.load_state_dict(payload["target_state_dict"])
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        self.learn_steps = int(payload.get("learn_steps", 0))
        if "agent_state" in payload:
            self.best_score = float(payload["agent_state"].get("best_score", self.best_score))

    @staticmethod
    def load_metadata(path: str) -> dict:
        """Read model metadata without constructing networks first."""
        payload = SnakeDQNAgent._torch_load(path, map_location="cpu")
        cfg_data = payload.get("cfg", {})
        payload_state_encoding = str(payload.get("state_encoding", cfg_data.get("state_encoding", STATE_ENCODING_INTEGER)))
        if payload_state_encoding == "compact11":
            payload_state_encoding = STATE_ENCODING_INTEGER
        hidden_layers = SnakeDQNAgent._hidden_layers_from_payload(payload, TrainConfig().hidden_layers)
        return {
            "board_size": int(payload.get("board_size", 20)),
            "state_encoding": payload_state_encoding,
            "action_space_size": int(payload.get("action_space_size", len(ACTIONS))),
            "hidden_layers": hidden_layers,
            "cfg": cfg_data,
        }
