from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def sanitize_tensor(tensor: torch.Tensor, limit: float = 100.0) -> torch.Tensor:
    """Replace non-finite values and clamp large finite spikes."""
    return torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "crelu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {act_name}")


class ActorCriticSequence(nn.Module):
    is_recurrent = False
    is_sequence = True

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_encoder_obs: int,
        latent_dim: int,
        encoder_hidden_dims: list[int] | tuple[int, ...],
        actor_hidden_dims: list[int] | tuple[int, ...],
        critic_hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        orthogonal_init: bool = False,
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.orthogonal_init = orthogonal_init
        self.latent_dim = latent_dim

        act = get_activation(activation)
        self.encoder = self._build_mlp(num_encoder_obs, list(encoder_hidden_dims), latent_dim, act, output_gain=0.01)
        self.actor = self._build_mlp(num_obs + latent_dim, list(actor_hidden_dims), num_actions, act, output_gain=0.01)
        self.critic = self._build_mlp(num_critic_obs, list(critic_hidden_dims), 1, act, output_gain=0.01)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        self.latent: torch.Tensor | None = None
        Normal.set_default_validate_args = False

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module,
        output_gain: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(last_dim, hidden_dim)
            if self.orthogonal_init:
                nn.init.orthogonal_(linear.weight, np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
            layers += [linear, activation]
            last_dim = hidden_dim
        linear = nn.Linear(last_dim, output_dim)
        if self.orthogonal_init:
            nn.init.orthogonal_(linear.weight, output_gain)
            nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        return nn.Sequential(*layers)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        return None

    def update_distribution(self, observations: torch.Tensor, observation_history: torch.Tensor) -> None:
        observations = sanitize_tensor(observations)
        observation_history = sanitize_tensor(observation_history)
        self.latent = self.encoder(observation_history)
        mean = self.actor(torch.cat((observations, self.latent.detach()), dim=-1))
        mean = sanitize_tensor(mean)
        std = sanitize_tensor(self.std, limit=5.0).clamp_min(1.0e-3)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations: torch.Tensor, observation_history: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent(self) -> torch.Tensor:
        return self.latent

    def act_inference(self, observations: torch.Tensor, observation_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations = sanitize_tensor(observations)
        observation_history = sanitize_tensor(observation_history)
        self.latent = self.encoder(observation_history)
        actions_mean = self.actor(torch.cat((observations, self.latent), dim=-1))
        actions_mean = sanitize_tensor(actions_mean)
        return actions_mean, self.latent

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        critic_observations = sanitize_tensor(critic_observations)
        return self.critic(critic_observations)

    def encode(self, observation_history: torch.Tensor) -> torch.Tensor:
        observation_history = sanitize_tensor(observation_history)
        return self.encoder(observation_history)
