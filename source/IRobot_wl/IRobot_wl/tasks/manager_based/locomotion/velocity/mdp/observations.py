from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def leg_angle(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Task-space leg angle (theta0) from joint positions using forward kinematics.

    Args:
        env: The environment.
        asset_cfg: The asset configuration.
        leg_joint_names: List of leg joint names in order [hip_l, knee_l, hip_r, knee_r].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].

    Returns:
        Leg angles theta0 for each leg, shape (num_envs, 2).
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get joint positions for leg joints
    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]

    # Build theta1 and theta2 per leg
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)

    _, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)
    return theta0


def leg_angle_dot(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    wheel_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Task-space leg angular velocity theta0_dot using the WL-Gym mirrored convention."""
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None or wheel_joint_names is None:
        return torch.zeros(env.num_envs, 2, device=env.device)

    leg_joint_indices = asset.find_joints(leg_joint_names)[0]
    wheel_joint_indices = asset.find_joints(wheel_joint_names)[0]
    state = compute_vmc_state(
        asset.data.joint_pos,
        asset.data.joint_vel,
        leg_joint_indices,
        wheel_joint_indices,
        l1,
        l2,
        offset,
        theta1_offset,
        theta2_offset,
    )
    return state["theta0_dot"]


def leg_length(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Task-space leg length (L0) from joint positions using forward kinematics.

    Args:
        env: The environment.
        asset_cfg: The asset configuration.
        leg_joint_names: List of leg joint names in order [hip_l, knee_l, hip_r, knee_r].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].

    Returns:
        Leg lengths L0 for each leg, shape (num_envs, 2).
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, 2, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]

    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)

    L0, _ = forward_kinematics(theta1, theta2, l1, l2, offset)
    return L0


def leg_length_dot(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    wheel_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Task-space leg length velocity L0_dot using the WL-Gym mirrored convention."""
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import compute_vmc_state

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None or wheel_joint_names is None:
        return torch.zeros(env.num_envs, 2, device=env.device)

    leg_joint_indices = asset.find_joints(leg_joint_names)[0]
    wheel_joint_indices = asset.find_joints(wheel_joint_names)[0]
    state = compute_vmc_state(
        asset.data.joint_pos,
        asset.data.joint_vel,
        leg_joint_indices,
        wheel_joint_indices,
        l1,
        l2,
        offset,
        theta1_offset,
        theta2_offset,
    )
    return state["L0_dot"]


def wheel_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute wheel joint positions matching the original WL-Gym observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def wheel_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute wheel joint velocities matching the original WL-Gym observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def previous_action(
    env: ManagerBasedEnv,
    action_name: str | None = None,
) -> torch.Tensor:
    """Previous action buffer from Isaac Lab's action manager."""
    if action_name is None or len(env.action_manager.active_terms) == 1:
        return env.action_manager.prev_action

    idx = 0
    for term_name, term_dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=False):
        if term_name == action_name:
            return env.action_manager.prev_action[:, idx : idx + term_dim]
        idx += term_dim
    raise ValueError(f"Unable to find action term '{action_name}' in action manager.")


def joint_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint accelerations from articulation data."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def wl_vmc_commands(
    env: ManagerBasedEnv,
    command_name: str,
    height_command: float = 0.25,
    lin_vel_scale: float = 2.0,
    ang_vel_scale: float = 0.25,
    height_scale: float = 5.0,
) -> torch.Tensor:
    """Old WL-Gym command observation layout: [lin_vel_x, ang_vel_yaw, base_height_cmd]."""
    commands = env.command_manager.get_command(command_name)
    obs = torch.stack(
        [
            commands[:, 0] * lin_vel_scale,
            commands[:, 2] * ang_vel_scale,
            torch.full_like(commands[:, 0], height_command) * height_scale,
        ],
        dim=1,
    )
    return obs


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
