# DQN model and agent implementation for Snake.
from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .utils import ACTIONS, REVERSE_DIRECTION, TrainConfig
except ImportError:
    from utils import ACTIONS, REVERSE_DIRECTION, TrainConfig


class QNetwork(nn.Module):
    """Feed-forward network where input is the entire flattened board."""

    def __init__(self, input_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SnakeDQNAgent:
    """DQN agent with replay buffer, target network, and epsilon-greedy policy."""

    def __init__(self, cfg: TrainConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.input_size = cfg.board_size * cfg.board_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(self.input_size, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(self.input_size, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=cfg.memory_size)

        self.epsilon = cfg.epsilon_start
        self.learn_steps = 0

    def valid_action_indices(self, current_direction: str) -> list[int]:
        blocked = REVERSE_DIRECTION[current_direction]
        return [idx for idx, action in enumerate(ACTIONS) if action != blocked]

    def select_action(self, state: np.ndarray, valid_indices: list[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(valid_indices)

        state_t = torch.from_numpy(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        # Argmax only over valid actions to avoid extra mask allocations.
        valid_q = q_values[valid_indices]
        best_in_valid = int(torch.argmax(valid_q).item())
        return int(valid_indices[best_in_valid])

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.cfg.batch_size:
            return None

        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.stack(states).astype(np.float32, copy=False)).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.from_numpy(np.stack(next_states).astype(np.float32, copy=False)).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select with policy net, evaluate with target net.
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + self.cfg.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "board_size": self.cfg.board_size,
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
            "cfg": payload.get("cfg", {}),
        }
