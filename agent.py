# DQN model and agent implementation for Snake.
from __future__ import annotations

from collections import deque
import random

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
        self.policy_net = MLPQNetwork(12, hidden_layers=hidden_layers, output_size=len(self.action_space)).to(self.device)
        self.target_net = MLPQNetwork(12, hidden_layers=hidden_layers, output_size=len(self.action_space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, str, bool, float]] = deque(maxlen=cfg.memory_size)

        self.epsilon = cfg.epsilon_start
        self.learn_steps = 0
        self.gamma = cfg.gamma
        self.batch_size_current = cfg.batch_size
        self.replay_warmup = cfg.replay_warmup
        self.soft_tau = cfg.soft_tau

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
        if len(self.memory) < self.replay_warmup:
            return None

        sample_size = min(len(self.memory), self.batch_size_current)
        batch = random.sample(self.memory, sample_size)
        states, actions, rewards, next_states, next_directions, dones, discounts = zip(*batch)

        states_t = torch.from_numpy(np.stack(states).astype(np.float32, copy=False)).to(self.device)
        actions_t = torch.from_numpy(np.fromiter(actions, dtype=np.int64, count=sample_size)).to(self.device)
        rewards_t = torch.from_numpy(np.fromiter(rewards, dtype=np.float32, count=sample_size)).to(self.device)
        next_states_t = torch.from_numpy(np.stack(next_states).astype(np.float32, copy=False)).to(self.device)
        dones_t = torch.from_numpy(np.fromiter(dones, dtype=np.float32, count=sample_size)).to(self.device)
        discounts_t = torch.from_numpy(np.fromiter(discounts, dtype=np.float32, count=sample_size)).to(self.device)

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

    def save(self, path: str) -> None:
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
        payload = torch.load(path, map_location=self.device)
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

        self.policy_net.load_state_dict(payload["state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        self.learn_steps = int(payload.get("learn_steps", 0))

    @staticmethod
    def load_metadata(path: str) -> dict:
        """Read model metadata without constructing networks first."""
        payload = torch.load(path, map_location="cpu")
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
