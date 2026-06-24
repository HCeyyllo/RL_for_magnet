from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DDPGConfig:
    actor_lr: float = 1.0e-4
    critic_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 1.0e-3
    batch_size: int = 128
    replay_size: int = 1_000_000
    ou_mu: float = 0.0
    ou_theta: float = 0.15
    ou_sigma: float = 0.20


class Actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observation, action], dim=-1))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        seed: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_observations[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self) -> int:
        return self.size


class OUNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.20,
        seed: int | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def reset(self) -> None:
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * self.rng.standard_normal(self.action_dim)
        self.state = (self.state + dx).astype(np.float32)
        return self.state


class DDPGAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DDPGConfig | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DDPGConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.actor = Actor(observation_dim, action_dim).to(self.device)
        self.actor_target = Actor(observation_dim, action_dim).to(self.device)
        self.critic = Critic(observation_dim, action_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.replay_buffer = ReplayBuffer(
            self.config.replay_size,
            observation_dim,
            action_dim,
            seed=seed,
        )
        self.noise = OUNoise(
            action_dim,
            mu=self.config.ou_mu,
            theta=self.config.ou_theta,
            sigma=self.config.ou_sigma,
            seed=seed,
        )

    def select_action(self, observation: np.ndarray, add_noise: bool = True) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(obs).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def select_actions(
        self,
        observations: np.ndarray,
        add_noise: bool = True,
        noises: list[OUNoise] | None = None,
    ) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            actions = self.actor(obs).cpu().numpy()
        self.actor.train()
        if add_noise:
            if noises is None:
                noise_values = np.stack([self.noise.sample() for _ in range(len(actions))])
            else:
                if len(noises) != len(actions):
                    raise ValueError("Number of noise processes must match number of observations.")
                noise_values = np.stack([noise.sample() for noise in noises])
            actions = actions + noise_values
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.config.batch_size,
            self.device,
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            y = rewards + self.config.gamma * (1.0 - dones) * target_q

        q = self.critic(obs, actions)
        critic_loss = F.mse_loss(q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load(self, path: str | Path, load_optimizers: bool = False) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * source_param.data)
