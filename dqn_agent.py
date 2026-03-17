from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class NStepReplayBuffer:
    """Replay buffer avec n-step returns.

    Pour chaque transition stockée, la reward est la somme actualisée sur n steps :
        R = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(n-1)*r_{n-1}
    et next_state est l'état au step n (pas step 1).
    Cela réduit la variance des estimations Q et accélère la propagation du signal.
    """

    def __init__(self, capacity: int, n_step: int, gamma: float) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self._pending: Deque[Transition] = deque(maxlen=n_step)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._pending.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=float(done),
            )
        )

        if len(self._pending) == self.n_step or done:
            self._flush_pending(force_done=bool(done))

        if done:
            # Vider le reste du pending buffer
            while self._pending:
                self._flush_pending(force_done=True)

    def _flush_pending(self, force_done: bool = False) -> None:
        if not self._pending:
            return
        first = self._pending[0]
        # Calcul de la reward n-step
        n_reward = 0.0
        gamma_pow = 1.0
        last_done = False
        last_next_state = first.next_state
        for i, t in enumerate(self._pending):
            n_reward += gamma_pow * t.reward
            gamma_pow *= self.gamma
            last_next_state = t.next_state
            if t.done:
                last_done = True
                break

        self.buffer.append(
            Transition(
                state=first.state,
                action=first.action,
                reward=n_reward,
                next_state=last_next_state,
                done=float(last_done or force_done),
            )
        )
        self._pending.popleft()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        n_step: int = 3,
        target_update_freq: int = 1,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        learning_starts: int = 1_000,
        double_dqn: bool = True,
        grad_clip: float = 1.0,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_starts = learning_starts
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        self.learn_step = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.SmoothL1Loss()

        if n_step > 1:
            self.replay_buffer: ReplayBuffer | NStepReplayBuffer = NStepReplayBuffer(
                capacity=buffer_capacity, n_step=n_step, gamma=gamma
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        self.q_net.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        self.q_net.train()
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # γ^n pour les n-step returns
        gamma_n = self.gamma ** self.n_step

        self.q_net.train()
        current_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            self.target_net.eval()
            if self.double_dqn:
                self.q_net.eval()
                next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                self.q_net.train()
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + gamma_n * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self._soft_update_target()

        return float(loss.item())

    def _soft_update_target(self) -> None:
        for target_param, source_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | None = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))
