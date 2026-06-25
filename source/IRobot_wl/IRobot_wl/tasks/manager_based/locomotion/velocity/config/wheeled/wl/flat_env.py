from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


class ClippedRewardManager(RewardManager):
    """Reward manager matching Wheel-Legged-Gym's per-term clipping."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv, clip_single_reward: float = 1.0):
        self.clip_single_reward = clip_single_reward
        super().__init__(cfg, env)

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        clip_value = self.clip_single_reward * dt

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.clamp(value, -clip_value, clip_value)
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf


class WLVMCFlatEnv(ManagerBasedRLEnv):
    """Flat VMC environment with reference-style reward clipping."""

    def load_managers(self):
        super().load_managers()
        self.reward_manager = ClippedRewardManager(self.cfg.rewards, self, clip_single_reward=1.0)

