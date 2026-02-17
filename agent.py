# DQN model and agent implementation for Snake.
from __future__ import annotations

from collections import deque
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .utils import (
        ACTIONS,
        REVERSE_DIRECTION,
        EpisodeDynamics,
        STATE_ENCODING_INTEGER,
        STATE_SIZE,
        SUPPORTED_STATE_ENCODINGS,
        TrainConfig,
        normalize_hidden_layers,
    )
except ImportError:
    from utils import (
        ACTIONS,
        REVERSE_DIRECTION,
        EpisodeDynamics,
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


class MLPQNetwork(nn.Module):
    """Feed-forward network for compact integer vector state."""

    def __init__(self, input_size: int, hidden_layers: tuple[int, ...], output_size: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty.")

        in_features = input_size
        first_width = hidden_layers[0]
        layers.append(nn.Linear(in_features, first_width))
        layers.append(nn.LayerNorm(first_width))
        layers.append(nn.ReLU())
        in_features = first_width

        for width in hidden_layers[1:]:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        hidden_layers = normalize_hidden_layers(cfg.hidden_layers)
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
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, str, bool, float]] = deque(maxlen=cfg.memory_size)

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

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
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
        batch = random.sample(self.memory, sample_size)

        states_np = np.empty((sample_size, STATE_SIZE), dtype=np.float32)
        actions_np = np.empty(sample_size, dtype=np.int64)
        rewards_np = np.empty(sample_size, dtype=np.float32)
        next_states_np = np.empty((sample_size, STATE_SIZE), dtype=np.float32)
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

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN with invalid-action masking on action selection.
            next_policy_q = self.policy_net(next_states_t)
            masked_next_policy_q = torch.full_like(next_policy_q, -1e9)
            for idx, direction in enumerate(next_directions):
                valid_indices = VALID_ACTIONS_BY_DIRECTION[direction]
                masked_next_policy_q[idx, valid_indices] = next_policy_q[idx, valid_indices]
            next_actions = masked_next_policy_q.argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + discounts_t * next_q * (1.0 - dones_t)
        td_error = target_q - current_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

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
            torch.random.set_rng_state(torch_cpu_state)
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def save_checkpoint(self, path: str, episode_index: int, replay_buffer) -> None:
        """Save full training state to allow exact resume."""
        replay_items = list(replay_buffer)
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

    def load_checkpoint(self, path: str) -> tuple[int, deque]:
        """Load full training state and return (episode_index, replay_buffer)."""
        payload = self._torch_load(path, map_location=self.device)
        # Backward compatibility: old saves only had model weights and metadata.
        if "policy_state_dict" not in payload or "optimizer_state_dict" not in payload:
            self.load(path)
            self.loaded_legacy_payload = True
            replay_buffer = deque(maxlen=self.cfg.memory_size)
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
