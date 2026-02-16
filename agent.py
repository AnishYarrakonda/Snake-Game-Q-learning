# DQN model and agent implementation for Snake.
from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .utils import ACTIONS, RELATIVE_ACTIONS, REVERSE_DIRECTION, TURN_LEFT, TURN_RIGHT, EpisodeDynamics, TrainConfig
except ImportError:
    from utils import ACTIONS, RELATIVE_ACTIONS, REVERSE_DIRECTION, TURN_LEFT, TURN_RIGHT, EpisodeDynamics, TrainConfig

VALID_ACTIONS_BY_DIRECTION = {
    direction: [idx for idx, action in enumerate(ACTIONS) if action != REVERSE_DIRECTION[direction]]
    for direction in ACTIONS
}


class MLPQNetwork(nn.Module):
    """Feed-forward network used for compact vector state."""

    def __init__(self, input_size: int, hidden_dim: int = 256, output_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvQNetwork(nn.Module):
    """Convolutional network for board tensor state."""

    def __init__(self, board_size: int, output_size: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        flattened = 64 * board_size * board_size
        self.head = nn.Sequential(
            nn.Linear(flattened, 256),
            nn.SiLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


class SnakeDQNAgent:
    """DQN agent with replay buffer, target network, and epsilon-greedy policy."""

    def __init__(self, cfg: TrainConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.uses_relative_actions = cfg.state_encoding == "compact11"
        self.action_space = RELATIVE_ACTIONS if self.uses_relative_actions else ACTIONS
        if device is not None:
            self.device = device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.uses_relative_actions:
            self.policy_net = MLPQNetwork(11, hidden_dim=cfg.hidden_dim, output_size=len(self.action_space)).to(self.device)
            self.target_net = MLPQNetwork(11, hidden_dim=cfg.hidden_dim, output_size=len(self.action_space)).to(self.device)
        else:
            self.policy_net = ConvQNetwork(cfg.board_size, output_size=len(self.action_space)).to(self.device)
            self.target_net = ConvQNetwork(cfg.board_size, output_size=len(self.action_space)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool, float]] = deque(maxlen=cfg.memory_size)

        self.epsilon = cfg.epsilon_start
        self.learn_steps = 0
        self.gamma = cfg.gamma
        self.long_term_weight = 1.0
        self.target_update_mode = "hard"
        self.target_update_every = cfg.target_update_every
        self.target_soft_tau = 0.0

    def valid_action_indices(self, current_direction: str) -> list[int]:
        if self.uses_relative_actions:
            return [0, 1, 2]
        return VALID_ACTIONS_BY_DIRECTION[current_direction]

    def action_to_direction(self, current_direction: str, action_idx: int) -> str:
        if not self.uses_relative_actions:
            return ACTIONS[action_idx]
        if action_idx == 0:
            return current_direction
        if action_idx == 1:
            return TURN_RIGHT[current_direction]
        return TURN_LEFT[current_direction]

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
        done: bool,
        discount: float,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done, discount))

    def apply_episode_dynamics(self, dynamics: EpisodeDynamics) -> None:
        self.gamma = dynamics.gamma
        self.long_term_weight = dynamics.long_term_weight
        self.epsilon = dynamics.epsilon
        self.target_update_mode = dynamics.target_update_mode
        self.target_update_every = dynamics.target_update_every
        self.target_soft_tau = dynamics.target_soft_tau
        for group in self.optimizer.param_groups:
            group["lr"] = dynamics.lr

    def train_step(self) -> float | None:
        if not self.memory:
            return None

        sample_size = min(len(self.memory), self.cfg.batch_size)
        batch = random.sample(self.memory, sample_size)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        states_t = torch.from_numpy(np.stack(states).astype(np.float32, copy=False)).to(self.device)
        actions_t = torch.from_numpy(np.fromiter(actions, dtype=np.int64, count=sample_size)).to(self.device)
        rewards_t = torch.from_numpy(np.fromiter(rewards, dtype=np.float32, count=sample_size)).to(self.device)
        next_states_t = torch.from_numpy(np.stack(next_states).astype(np.float32, copy=False)).to(self.device)
        dones_t = torch.from_numpy(np.fromiter(dones, dtype=np.float32, count=sample_size)).to(self.device)
        discounts_t = torch.from_numpy(np.fromiter(discounts, dtype=np.float32, count=sample_size)).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select with policy net, evaluate with target net.
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + self.long_term_weight * discounts_t * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.target_update_mode == "soft":
            tau = self.target_soft_tau
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - tau).add_(policy_param.data, alpha=tau)
        elif self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "board_size": self.cfg.board_size,
            "state_encoding": self.cfg.state_encoding,
            "action_space_size": len(self.action_space),
            "hidden_dim": self.cfg.hidden_dim,
            "state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "cfg": vars(self.cfg),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        board_size = int(payload.get("board_size", self.cfg.board_size))
        if board_size != self.cfg.board_size:
            raise ValueError(
                f"Model board size ({board_size}) does not match current board size ({self.cfg.board_size})."
            )
        payload_state_encoding = str(payload.get("state_encoding", payload.get("cfg", {}).get("state_encoding", "board")))
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

        self.policy_net.load_state_dict(payload["state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_start))
        self.learn_steps = int(payload.get("learn_steps", 0))

    @staticmethod
    def load_metadata(path: str) -> dict:
        """Read model metadata without constructing networks first."""
        payload = torch.load(path, map_location="cpu")
        return {
            "board_size": int(payload.get("board_size", 20)),
            "state_encoding": payload.get("state_encoding", payload.get("cfg", {}).get("state_encoding", "board")),
            "action_space_size": int(payload.get("action_space_size", len(ACTIONS))),
            "cfg": payload.get("cfg", {}),
        }
