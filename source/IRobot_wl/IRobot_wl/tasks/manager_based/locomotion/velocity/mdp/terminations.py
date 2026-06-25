from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sustained_flat_failure(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 10.0,
    projected_gravity_z_threshold: float = -0.1,
    hold_time_s: float = 1.0,
) -> torch.Tensor:
    """Terminate only when base contact or severe tilt persists continuously."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    base_contact = torch.any(
        torch.max(torch.norm(forces, dim=-1), dim=1)[0] > contact_threshold,
        dim=1,
    )
    fallen = env.scene["robot"].data.projected_gravity_b[:, 2] > projected_gravity_z_threshold
    failed = base_contact | fallen

    counter_name = "_wl_flat_failure_steps"
    counter = getattr(env, counter_name, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    counter = torch.where(failed, counter + 1, torch.zeros_like(counter))
    counter = torch.where(env.episode_length_buf <= 1, torch.zeros_like(counter), counter)
    setattr(env, counter_name, counter)

    required_steps = max(1, int(round(hold_time_s / env.step_dt)))
    return counter > required_steps
