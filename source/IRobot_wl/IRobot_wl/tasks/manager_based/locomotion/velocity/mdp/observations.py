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
    """Forward-positive wheel joint positions for both wheels."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()
    wheel_pos[:, 0] = -wheel_pos[:, 0]
    return wheel_pos


def wheel_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward-positive wheel joint velocities for both wheels."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids].clone()
    wheel_vel[:, 0] = -wheel_vel[:, 0]
    return wheel_vel


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


def previous_previous_action(
    env: ManagerBasedEnv,
    action_name: str | None = None,
) -> torch.Tensor:
    """Two-step action history from the action term when available."""
    if action_name is not None:
        term = env.action_manager.get_term(action_name)
        if hasattr(term, "previous_previous_actions"):
            return term.previous_previous_actions
    if hasattr(env.action_manager, "prev_prev_action"):
        return env.action_manager.prev_prev_action
    return torch.zeros_like(previous_action(env, action_name))


def joint_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint accelerations from articulation data."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def applied_joint_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Applied joint torques for privileged critic observations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def body_mass_delta(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current body mass minus default mass, summed over selected bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        masses = asset.root_physx_view.get_masses().to(env.device)
        default_masses = asset.data.default_mass.to(env.device)
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            delta = masses[:, body_ids] - default_masses[:, body_ids]
        else:
            delta = masses[:, body_ids] - default_masses[:, body_ids]
        return torch.sum(delta, dim=1, keepdim=True)
    except Exception:
        return torch.zeros(env.num_envs, 1, device=env.device)


def body_com_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current local body center-of-mass position."""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        coms = asset.root_physx_view.get_coms().to(env.device)[..., :3]
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            selected = coms[:, body_ids]
        else:
            selected = coms[:, body_ids]
        return torch.mean(selected, dim=1)
    except Exception:
        return torch.zeros(env.num_envs, 3, device=env.device)


def default_joint_pos_delta(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Delta between randomized default joint position and the first observed default."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(default_joint_pos_delta, "_raw_defaults"):
        default_joint_pos_delta._raw_defaults = {}
    env_key = id(env)
    if env_key not in default_joint_pos_delta._raw_defaults:
        default_joint_pos_delta._raw_defaults[env_key] = asset.data.default_joint_pos.clone()
    raw_default = default_joint_pos_delta._raw_defaults[env_key]
    return asset.data.default_joint_pos[:, asset_cfg.joint_ids] - raw_default[:, asset_cfg.joint_ids]


def material_static_friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean static friction coefficient over selected robot collision shapes."""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        materials = asset.root_physx_view.get_material_properties().to(env.device)
        return torch.mean(materials[..., 0], dim=1, keepdim=True)
    except Exception:
        return torch.zeros(env.num_envs, 1, device=env.device)


def material_restitution(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean restitution coefficient over selected robot collision shapes."""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        materials = asset.root_physx_view.get_material_properties().to(env.device)
        return torch.mean(materials[..., 2], dim=1, keepdim=True)
    except Exception:
        return torch.zeros(env.num_envs, 1, device=env.device)


def wl_vmc_commands(
    env: ManagerBasedEnv,
    command_name: str,
    height_command: float = 0.25,
    lin_vel_scale: float = 2.0,
    ang_vel_scale: float = 0.25,
    height_scale: float = 5.0,
) -> torch.Tensor:
    """WL-Gym command observation layout: [lin_vel_x, ang_vel_yaw, base_height_cmd].

    Height commands are randomized in [0.1, 0.25] and resampled every 5 seconds,
    matching the original WL-Gym behaviour.
    """
    commands = env.command_manager.get_command(command_name)

    # Height command with periodic resampling (matching WL-Gym resampling_time = 5s)
    resample_every = int(5.0 / env.step_dt)
    resample_mask = (env.episode_length_buf % resample_every) == 0

    # Use function attribute to persist height buffer across calls
    if not hasattr(wl_vmc_commands, "_height_buf"):
        wl_vmc_commands._height_buf = {}  # keyed by id(env) to handle multiple envs
    env_key = id(env)
    if env_key not in wl_vmc_commands._height_buf:
        wl_vmc_commands._height_buf[env_key] = (
            0.1 + 0.15 * torch.rand(env.num_envs, device=env.device)
        )

    height_buf = wl_vmc_commands._height_buf[env_key]
    height_buf[resample_mask] = 0.1 + 0.15 * torch.rand(int(resample_mask.sum()), device=env.device)

    obs = torch.stack(
        [
            commands[:, 0] * lin_vel_scale,
            commands[:, 2] * ang_vel_scale,
            height_buf * height_scale,
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
