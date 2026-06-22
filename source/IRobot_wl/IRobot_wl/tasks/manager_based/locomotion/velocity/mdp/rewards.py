from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_z_cmd_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize squared yaw-rate tracking error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    reward = torch.square(error)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    lin_x_threshold: float | None = None,
    ang_z_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from default joint positions only for true standing commands."""
    command = env.command_manager.get_command(command_name)
    if lin_x_threshold is None:
        lin_x_threshold = command_threshold
    if ang_z_threshold is None:
        ang_z_threshold = command_threshold
    is_standing_command = (torch.abs(command[:, 0]) < lin_x_threshold) & (torch.abs(command[:, 2]) < ang_z_threshold)
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= is_standing_command
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(
#         env.action_manager.action - 2 * env.action_manager.prev_action
#         + env.action_manager.prev_prev_action
#     )
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def self_right_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ungated squared error from projected gravity to [0, 0, -1]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.tensor([0.0, 0.0, -1.0], device=asset.data.projected_gravity_b.device)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)


def self_right_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ungated vertical base velocity penalty for recovery tasks."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def self_right_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ungated roll/pitch angular velocity penalty for recovery tasks."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def self_right_body_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ungated body linear acceleration penalty for recovery tasks."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if hasattr(asset.data, "body_acc_w"):
        return torch.sum(torch.square(asset.data.body_acc_w[:, asset_cfg.body_ids, :3]), dim=(1, 2))
    return torch.zeros(env.num_envs, device=env.device)


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant alive bonus."""
    return torch.ones(env.num_envs, device=env.device)


# --------------------------------------------------------------------------- #
# VMC / Wheel-Legged-Gym reward functions
# --------------------------------------------------------------------------- #


def nominal_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Penalize asymmetry between left and right leg angles (VMC key reward).

    This reward encourages the robot to keep its legs symmetric, which promotes
    stable standing and coordinated locomotion.

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration.
        leg_joint_names: Leg joint names [hip_l, knee_l, hip_r, knee_r].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].

    Returns:
        Reward value, shape (num_envs,).
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]

    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)

    _, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)

    # Positive penalty magnitude: 0 when symmetric, larger when left/right diverge.
    ang_diff = torch.square(theta0[:, 0] - theta0[:, 1])
    reward = ang_diff
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def leg_length_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Penalize left/right leg-length mismatch in task space."""
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    leg_length, _ = forward_kinematics(theta1, theta2, l1, l2, offset)
    reward = torch.square(leg_length[:, 0] - leg_length[:, 1])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def leg_angle_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Penalize left/right leg-angle mismatch in task space."""
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    _, leg_angle = forward_kinematics(theta1, theta2, l1, l2, offset)
    reward = torch.square(leg_angle[:, 0] - leg_angle[:, 1])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def theta0_nominal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
) -> torch.Tensor:
    """Penalize leg-angle deviation from vertical (θ0 ≈ 0) in task space.

    Unlike ``joint_pos_penalty`` which works in joint space and implicitly
    couples θ0 with L0, this reward only constrains the leg direction — the
    policy remains free to choose knee configuration for the desired leg length.

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration.
        leg_joint_names: Leg joint names [hip_l, knee_l, hip_r, knee_r].
        l1: Thigh link length [m].
        l2: Calf link length [m].
        offset: Hip offset from body center [m].
        theta1_offset: Hip zero-angle offset [rad].
        theta2_offset: Knee-to-wheel zero-angle offset [rad].

    Returns:
        Penalty value, shape (num_envs,). 0 when both legs point straight down.
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    _, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)

    reward = torch.sum(torch.square(theta0), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def vmc_action_symmetry(
    env: ManagerBasedRLEnv,
    action_name: str = "vmc",
    tp_scale: float = 1.0,
    force_scale: float = 1.0,
    wheel_scale: float = 0.25,
) -> torch.Tensor:
    """Penalize left/right task-space action mismatch for symmetric gliding."""
    if action_name is None:
        actions = env.action_manager.action
    else:
        action_term = env.action_manager.get_term(action_name)
        actions = getattr(action_term, "processed_actions", action_term.raw_actions)
    tp_diff = torch.square(actions[:, 0] - actions[:, 3]) * tp_scale
    force_diff = torch.square(actions[:, 1] - actions[:, 4]) * force_scale
    wheel_diff = torch.square(actions[:, 2] - actions[:, 5]) * wheel_scale
    reward = tp_diff + force_diff + wheel_diff
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_enhance(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Enhanced base height tracking using exponential kernel.

    A tighter version of base_height_l2 that uses an exponential kernel
    for more precise height tracking, matching WL-Gym's approach.

    Args:
        env: The RL environment.
        target_height: Desired base height [m].
        asset_cfg: The asset configuration.
        sensor_cfg: Optional height scanner configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height

    base_height = torch.mean(
        asset.data.root_pos_w[:, 2].unsqueeze(1) - 0.0, dim=1
    )  # simplified for flat terrain
    base_height_error = torch.square(base_height - adjusted_target_height)
    reward = torch.exp(-base_height_error / 0.001 / 10) - 1  # WL-Gym style
    return reward


def command_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float | None = None,
    fallback_target_height: float = 0.19,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize root height error from the sampled base-height command."""
    command_term = env.command_manager.get_term(command_name)
    sampled_target_height = getattr(command_term, "base_height_command_b", None)
    if sampled_target_height is None:
        fallback = fallback_target_height if target_height is None else target_height
        sampled_target_height = torch.full((env.num_envs,), fallback, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = sampled_target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = sampled_target_height

    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def command_base_height_enhance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float | None = None,
    fallback_target_height: float = 0.19,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Exponential base-height tracking reward using the sampled height command."""
    command_term = env.command_manager.get_term(command_name)
    sampled_target_height = getattr(command_term, "base_height_command_b", None)
    if sampled_target_height is None:
        fallback = fallback_target_height if target_height is None else target_height
        sampled_target_height = torch.full((env.num_envs,), fallback, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = sampled_target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = sampled_target_height

    base_height = torch.mean(asset.data.root_pos_w[:, 2].unsqueeze(1) - 0.0, dim=1)
    base_height_error = torch.square(base_height - adjusted_target_height)
    return torch.exp(-base_height_error / 0.001 / 10) - 1


def command_base_height_over_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin: float = 0.015,
    target_height: float | None = None,
    fallback_target_height: float = 0.19,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize only base height above the sampled command plus a margin."""
    command_term = env.command_manager.get_term(command_name)
    sampled_target_height = getattr(command_term, "base_height_command_b", None)
    if sampled_target_height is None:
        fallback = fallback_target_height if target_height is None else target_height
        sampled_target_height = torch.full((env.num_envs,), fallback, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = sampled_target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = sampled_target_height

    over_height = torch.clamp(asset.data.root_pos_w[:, 2] - adjusted_target_height - margin, min=0.0)
    reward = torch.square(over_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def vmc_force_action_l2(env: ManagerBasedRLEnv, action_name: str = "vmc") -> torch.Tensor:
    """Penalize sustained use of the two VMC axial-force action channels."""
    action_term = env.action_manager.get_term(action_name)
    actions = getattr(action_term, "processed_actions", action_term.raw_actions)
    return torch.sum(torch.square(actions[:, [1, 4]]), dim=1)


def vmc_force_over_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "vmc",
    force_limit_n: float = 65.0,
    normalize_n: float = 20.0,
) -> torch.Tensor:
    """Penalize axial support force only above a reasonable per-leg limit."""
    action_term = env.action_manager.get_term(action_name)
    actions = getattr(action_term, "processed_actions", action_term.raw_actions)
    force_actions = actions[:, [1, 4]]

    action_scale_force = getattr(action_term.cfg, "action_scale_force", 1.0)
    feedforward_force = getattr(action_term.cfg, "feedforward_force", 0.0)
    force_leg = force_actions * action_scale_force + feedforward_force
    over_force = torch.clamp(force_leg - force_limit_n, min=0.0) / normalize_n
    reward = torch.sum(torch.square(over_force), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def body_lin_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear acceleration using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # body_acc = (current_vel - last_vel) / dt is approximately available
    # Use body_lin_acc computed from root_acceleration if available
    if hasattr(asset.data, "body_acc_w"):
        reward = torch.sum(torch.square(asset.data.body_acc_w[:, asset_cfg.body_ids, :3]), dim=(1, 2))
    else:
        reward = torch.zeros(env.num_envs, device=env.device)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize collisions on selected bodies (WL-Gym style).

    Args:
        env: The RL environment.
        sensor_cfg: Contact sensor configuration for bodies to monitor.
        threshold: Contact force threshold for collision detection [N].

    Returns:
        Reward value, shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_collision = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0] > threshold
    reward = torch.sum(is_collision, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg joint velocities (WL-Gym style, only hip joints).

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration (should specify hip joint names).

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return reward


def track_lin_vel_enhance(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Enhanced tracking of linear velocity commands (X-direction only).

    Uses a sharper exponential kernel (sigma/10) for precise velocity tracking,
    matching the original WL-Gym ``_reward_tracking_lin_vel_enhance``.

    Args:
        env: The RL environment.
        std: Base tracking standard deviation. The enhance kernel uses std/10.
        command_name: The name of the command term.
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-lin_vel_error / (std / 10)) - 1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_enhance(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Enhanced tracking of angular velocity commands (yaw only).

    Uses a sharper exponential kernel (sigma/10) for precise yaw rate tracking,
    matching the original WL-Gym ``_reward_tracking_ang_vel_enhance``.

    Args:
        env: The RL environment.
        std: Base tracking standard deviation. The enhance kernel uses std/10.
        command_name: The name of the command term.
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / (std / 10)) - 1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_smooth(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Second-order action smoothness penalty for leg actions (WL-Gym style).

    Penalizes the second derivative of leg actions (indices 0,1 for left leg
    and 3,4 for right leg), matching the original WL-Gym
    ``_reward_action_smooth``. Wheel velocity actions (indices 2, 5) are
    excluded from this penalty.

    Since Isaac Lab's ActionManager only tracks one step of previous actions,
    this function manually maintains a ``prev_prev_action`` buffer via function
    attributes.

    Returns:
        Reward value, shape (num_envs,).
    """
    actions = env.action_manager.action
    prev = env.action_manager.prev_action

    # Manually track prev_prev_action (Isaac Lab only keeps one step of history)
    if not hasattr(action_smooth, "_prev_prev_buf"):
        action_smooth._prev_prev_buf = {}
    env_key = id(env)
    if env_key not in action_smooth._prev_prev_buf:
        action_smooth._prev_prev_buf[env_key] = torch.zeros_like(prev)
    prev_prev = action_smooth._prev_prev_buf[env_key]

    # Only apply to leg actions: indices [0,1] (left theta, L0) and [3,4] (right theta, L0)
    diff = torch.sum(
        torch.square(actions[:, :2] - 2 * prev[:, :2] + prev_prev[:, :2]), dim=1
    ) + torch.sum(
        torch.square(actions[:, 3:5] - 2 * prev[:, 3:5] + prev_prev[:, 3:5]), dim=1
    )

    # Shift history: prev becomes prev_prev for next call
    action_smooth._prev_prev_buf[env_key] = prev.clone()

    # Ignore first steps
    mask = (prev[:, 0] != 0).float() * (prev_prev[:, 0] != 0).float()
    diff = diff * mask
    return diff


def torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Penalize torques that are too close to the limit (WL-Gym style).

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration.
        soft_ratio: Ratio of the limit at which penalty begins.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_limits = asset.data.torque_limit[:, asset_cfg.joint_ids]
    out_of_limits = (torch.abs(torques) - torque_limits * soft_ratio).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# --------------------------------------------------------------------------- #
# Stand-up recovery reward functions
# --------------------------------------------------------------------------- #


def is_upright(env: ManagerBasedRLEnv, threshold: float = -0.7) -> torch.Tensor:
    """Boolean mask: True when the robot is upright enough for locomotion.

    Args:
        env: The RL environment.
        threshold: projected_gravity_z threshold below which robot is "upright".

    Returns:
        Boolean tensor, shape (num_envs,).
    """
    return env.scene["robot"].data.projected_gravity_b[:, 2] < threshold


def is_fallen(env: ManagerBasedRLEnv, threshold: float = -0.3) -> torch.Tensor:
    """Boolean mask: True when the robot is clearly fallen and needs recovery.

    Args:
        env: The RL environment.
        threshold: projected_gravity_z threshold above which robot is "fallen".

    Returns:
        Boolean tensor, shape (num_envs,).
    """
    return env.scene["robot"].data.projected_gravity_b[:, 2] > threshold


def recovery_upright_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward progress toward upright posture during recovery.

    Uses exponential kernel on the deviation of projected_gravity_z from -1
    (perfectly upright). Stronger reward as the robot gets closer to upright.

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_z = -1 means perfectly upright, 0 means horizontal, 1 means upside-down
    # We want it as close to -1 as possible
    upright_error = torch.square(asset.data.projected_gravity_b[:, 2] + 1.0)
    reward = torch.exp(-upright_error / 0.1)  # std=sqrt(0.1) ≈ 0.316, sensitive kernel
    return reward


def self_right_attitude(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body attitude approaching projected gravity [0, 0, -1]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    upright_error = torch.sum(torch.square(gravity_b[:, :2]), dim=1) + torch.square(gravity_b[:, 2] + 1.0)
    return torch.exp(-upright_error / (std**2))


def self_right_tilt_progress(
    env: ManagerBasedRLEnv,
    max_reward: float = 0.05,
    max_penalty: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward step-to-step reduction in body-frame gravity xy tilt."""
    asset: RigidObject = env.scene[asset_cfg.name]
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)

    prev_name = "_self_right_prev_tilt_xy"
    prev_tilt = getattr(env, prev_name, None)
    if prev_tilt is None or prev_tilt.shape != tilt.shape:
        prev_tilt = tilt.detach().clone()

    progress = prev_tilt - tilt
    if hasattr(env, "episode_length_buf"):
        progress = torch.where(env.episode_length_buf <= 1, torch.zeros_like(progress), progress)

    setattr(env, prev_name, tilt.detach().clone())
    return torch.clamp(progress, min=-max_penalty, max=max_reward)


def self_right_upright_success(
    env: ManagerBasedRLEnv,
    threshold: float = -0.85,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse bonus once the body is close to upright."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b[:, 2] < threshold).float()


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.19,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward increasing base height during stand-up recovery.

    When the robot is fallen, the base is near the ground. This reward
    encourages lifting the body up toward the target standing height.

    Args:
        env: The RL environment.
        target_height: Target standing base height [m].
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    # Normalize: 1.0 at target height, 0.0 at ground level
    height_ratio = torch.clamp(base_height / target_height, min=0.0, max=1.0)
    # Exponential reward: strong push when very low, saturates near target
    reward = 1.0 - torch.exp(-5.0 * height_ratio)
    return reward


def recovery_leg_extension(
    env: ManagerBasedRLEnv,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    l0_min: float = 0.1219258562330587,
    l0_max: float = 0.3006386827708927,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward extending legs to push the body up during stand-up recovery.

    Longer legs help lift the body off the ground. This reward encourages
    the policy to extend legs toward their maximum length.

    Args:
        env: The RL environment.
        leg_joint_names: Leg joint names [hip_l, knee_l, hip_r, knee_r].
        l1, l2, offset: Kinematic parameters.
        theta1_offset, theta2_offset: Joint angle offsets.
        l0_min, l0_max: Reachable leg length bounds.
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    L0, _ = forward_kinematics(theta1, theta2, l1, l2, offset)

    # Normalize leg length: 0 = min, 1 = max
    L0_norm = torch.clamp((L0 - l0_min) / (l0_max - l0_min), min=0.0, max=1.0)
    # Mean across both legs, encourage extension
    reward = torch.mean(L0_norm, dim=1)
    return reward


def recovery_leg_symmetry(
    env: ManagerBasedRLEnv,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize left/right leg asymmetry during stand-up recovery.

    Symmetric leg usage is crucial for stable recovery without tipping over.

    Args:
        env: The RL environment.
        leg_joint_names: Leg joint names.
        l1, l2, offset: Kinematic parameters.
        theta1_offset, theta2_offset: Joint angle offsets.
        asset_cfg: The asset configuration.

    Returns:
        Penalty value, shape (num_envs,). Higher = more asymmetric (less reward).
    """
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    asset: Articulation = env.scene[asset_cfg.name]
    if leg_joint_names is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    L0, theta0 = forward_kinematics(theta1, theta2, l1, l2, offset)

    # Combined asymmetry in both leg length and leg angle
    length_asym = torch.square(L0[:, 0] - L0[:, 1])
    angle_asym = torch.square(theta0[:, 0] - theta0[:, 1])
    return length_asym + angle_asym


def recovery_action_smoothness(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize jerky/aggressive actions during stand-up recovery.

    Smooth actions are more energy-efficient and reduce risk of damage.
    This is the same as action_rate_l2 but specifically scaled for recovery.

    Returns:
        Penalty value, shape (num_envs,).
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def recovery_stand_upright_factor(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth phase factor for stand-up rewards.

    Returns 0 when clearly fallen and 1 when close to upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    proj_z = asset.data.projected_gravity_b[:, 2]
    return torch.clamp((fallen_threshold - proj_z) / (fallen_threshold - upright_threshold), min=0.0, max=1.0)


def _recovery_stand_time_gate(
    env: ManagerBasedRLEnv,
    start_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Smoothly enable late-stage penalties after the policy has had time to act."""
    elapsed_time = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    if ramp_time_s <= 0.0:
        return (elapsed_time >= start_time_s).float()
    return torch.clamp((elapsed_time - start_time_s) / ramp_time_s, min=0.0, max=1.0)


def _recovery_stand_leg_state(
    env: ManagerBasedRLEnv,
    leg_joint_names: list[str] | None,
    l1: float,
    l2: float,
    offset: float,
    theta1_offset: float,
    theta2_offset: float,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return task-space leg length and swing angle for RecoveryStand rewards."""
    from IRobot_wl.tasks.manager_based.locomotion.velocity.mdp.vmc import forward_kinematics

    if leg_joint_names is None:
        zeros = torch.zeros(env.num_envs, 2, device=env.device)
        return zeros, zeros

    asset: Articulation = env.scene[asset_cfg.name]
    joint_indices = asset.find_joints(leg_joint_names)[0]
    dof_pos = asset.data.joint_pos[:, joint_indices]
    theta1 = torch.stack([dof_pos[:, 0] + theta1_offset, -dof_pos[:, 2] + theta1_offset], dim=1)
    theta2 = torch.stack([dof_pos[:, 1] + theta2_offset, -dof_pos[:, 3] + theta2_offset], dim=1)
    return forward_kinematics(theta1, theta2, l1, l2, offset)


def recovery_stand_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.18,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    gate_by_theta0: bool = False,
    theta0_ready_std: float = 0.35,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height error after upright, optionally only when wheels are under the body."""
    asset: RigidObject = env.scene[asset_cfg.name]
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    if gate_by_theta0:
        _, theta0 = _recovery_stand_leg_state(
            env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
        )
        theta0_ready = torch.exp(-torch.mean(torch.square(theta0), dim=1) / (theta0_ready_std**2))
        phase = phase * theta0_ready
    return phase * torch.square(asset.data.root_pos_w[:, 2] - target_height)


def recovery_stand_base_height_progress(
    env: ManagerBasedRLEnv,
    target_height: float = 0.19,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    theta0_ready_std: float = 0.35,
    max_progress_rate: float = 0.08,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward base-height progress once upright and leg angle is ready."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    prev_name = "_recovery_stand_prev_base_height"
    prev_height = getattr(env, prev_name, None)
    if prev_height is None or prev_height.shape != base_height.shape:
        prev_height = base_height.detach().clone()

    progress_rate = (base_height - prev_height) / env.step_dt
    progress_rate = torch.clamp(progress_rate, min=0.0, max=max_progress_rate)
    below_target = (base_height < target_height).float()

    _, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )
    theta0_ready = torch.exp(-torch.mean(torch.square(theta0), dim=1) / (theta0_ready_std**2))
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)

    setattr(env, prev_name, base_height.detach().clone())
    return phase * theta0_ready * below_target * progress_rate


def recovery_stand_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal drift after self-righting to encourage in-place stand-up."""
    asset: RigidObject = env.scene[asset_cfg.name]
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def recovery_stand_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw spinning after self-righting."""
    asset: RigidObject = env.scene[asset_cfg.name]
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * torch.square(asset.data.root_ang_vel_b[:, 2])


def recovery_stand_leg_length_l2(
    env: ManagerBasedRLEnv,
    retracted_length: float = 0.14,
    standing_length: float = 0.22,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phase-aware leg length target: retract while fallen, extend after upright."""
    leg_length, _ = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg).unsqueeze(1)
    target = retracted_length * (1.0 - phase) + standing_length * phase
    return torch.mean(torch.square(leg_length - target), dim=1)


def recovery_stand_leg_symmetry_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.75,
    fallen_threshold: float = -0.25,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Coordinate left/right leg length and swing angle after self-righting starts."""
    leg_length, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    length_error = torch.square(leg_length[:, 0] - leg_length[:, 1])
    theta_error = torch.square(theta0[:, 0] - theta0[:, 1])
    return phase * (length_error + 0.25 * theta_error)


def recovery_stand_theta0_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bring the wheels under the body after the robot is mostly upright."""
    _, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * torch.mean(torch.square(theta0), dim=1)


def recovery_stand_theta0_worst_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the worst leg swing angle after the body starts to recover."""
    _, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * torch.max(torch.square(theta0), dim=1).values


def recovery_stand_splayed_long_leg_l2(
    env: ManagerBasedRLEnv,
    theta0_threshold: float = 0.8,
    leg_length_threshold: float = 0.18,
    upright_threshold: float = -0.7,
    fallen_threshold: float = -0.25,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Discourage the stable failure mode with long legs splayed away from the body."""
    leg_length, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg).unsqueeze(1)
    theta_excess = torch.clamp(torch.abs(theta0) - theta0_threshold, min=0.0)
    length_excess = torch.clamp(leg_length - leg_length_threshold, min=0.0)
    return torch.sum(phase * torch.square(theta_excess) * torch.square(length_excess / 0.05), dim=1)


def recovery_stand_theta0_ready_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward placing both wheels below the body before asking for full lift."""
    _, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    theta0_error = torch.mean(torch.square(theta0), dim=1)
    return phase * torch.exp(-theta0_error / (std**2))


def recovery_stand_negative_force_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "vmc",
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Discourage unloading the legs after the body is upright."""
    action_term = env.action_manager.get_term(action_name)
    actions = getattr(action_term, "processed_actions", action_term.raw_actions)
    negative_force_actions = torch.clamp(-actions[:, [1, 4]], min=0.0)

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg).unsqueeze(1)
    return torch.mean(torch.square(negative_force_actions) * phase, dim=1)


def recovery_stand_leg_ground_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg-link ground contact during the upright stand-up phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    contact_count = torch.sum((contact_force > threshold).float(), dim=1)

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * contact_count


def recovery_stand_wheel_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward wheel-ground contact after self-righting."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    in_contact = contact_force > threshold
    mean_contact = torch.mean(in_contact.float(), dim=1)
    both_contact = torch.all(in_contact, dim=1).float()

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * (0.5 * mean_contact + 0.5 * both_contact)


def recovery_stand_wheel_load(
    env: ManagerBasedRLEnv,
    target_total_force_n: float = 100.0,
    min_each_force_n: float = 25.0,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the wheels carrying body weight after self-righting."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    wheel_force_z = torch.max(torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2]), dim=1)[0]
    total_load = torch.sum(wheel_force_z, dim=1)

    total_load_score = torch.clamp(total_load / target_total_force_n, min=0.0, max=1.0)
    each_load_score = torch.mean(torch.clamp(wheel_force_z / min_each_force_n, min=0.0, max=1.0), dim=1)

    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * (0.6 * total_load_score + 0.4 * each_load_score)


def recovery_stand_late_not_upright(
    env: ManagerBasedRLEnv,
    start_time_s: float = 4.0,
    ramp_time_s: float = 1.5,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize episodes that have not self-righted after the early recovery window."""
    time_gate = _recovery_stand_time_gate(env, start_time_s, ramp_time_s)
    upright = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return time_gate * torch.square(1.0 - upright)


def recovery_stand_late_low_base(
    env: ManagerBasedRLEnv,
    min_height: float = 0.17,
    height_std: float = 0.04,
    start_time_s: float = 7.0,
    ramp_time_s: float = 2.0,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize staying in a low crouch after the body is already mostly upright."""
    asset: RigidObject = env.scene[asset_cfg.name]
    time_gate = _recovery_stand_time_gate(env, start_time_s, ramp_time_s)
    upright = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    height_deficit = torch.clamp(min_height - asset.data.root_pos_w[:, 2], min=0.0)
    return time_gate * upright * torch.square(height_deficit / height_std)


def recovery_stand_late_no_wheel_load(
    env: ManagerBasedRLEnv,
    target_total_force_n: float = 100.0,
    min_each_force_n: float = 25.0,
    min_load_score: float = 0.65,
    start_time_s: float = 7.0,
    ramp_time_s: float = 2.0,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize upright poses where the wheels are not carrying enough weight."""
    load_score = recovery_stand_wheel_load(
        env,
        target_total_force_n=target_total_force_n,
        min_each_force_n=min_each_force_n,
        upright_threshold=upright_threshold,
        fallen_threshold=fallen_threshold,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    time_gate = _recovery_stand_time_gate(env, start_time_s, ramp_time_s)
    return time_gate * torch.square(torch.clamp(min_load_score - load_score, min=0.0))


def recovery_stand_timeout_not_upright(
    env: ManagerBasedRLEnv,
    max_time_s: float = 5.0,
    min_upright_factor: float = 0.55,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when self-righting has clearly failed within the early window."""
    elapsed_time = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    upright = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return (elapsed_time > max_time_s) & (upright < min_upright_factor)


def recovery_stand_timeout_not_standing(
    env: ManagerBasedRLEnv,
    max_time_s: float = 9.0,
    min_upright_factor: float = 0.75,
    min_height: float = 0.165,
    target_total_force_n: float = 100.0,
    min_each_force_n: float = 25.0,
    min_load_score: float = 0.35,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate late episodes that are upright-ish but not truly standing on the wheels."""
    asset: RigidObject = env.scene[asset_cfg.name]
    elapsed_time = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    upright = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    load_score = recovery_stand_wheel_load(
        env,
        target_total_force_n=target_total_force_n,
        min_each_force_n=min_each_force_n,
        upright_threshold=upright_threshold,
        fallen_threshold=fallen_threshold,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )

    not_standing = (upright < min_upright_factor) | (asset.data.root_pos_w[:, 2] < min_height) | (
        load_score < min_load_score
    )
    return (elapsed_time > max_time_s) & not_standing


def recovery_stand_success_bonus(
    env: ManagerBasedRLEnv,
    target_height: float = 0.19,
    min_leg_length: float = 0.145,
    theta0_threshold: float = 0.35,
    height_margin: float = 0.015,
    lin_vel_threshold: float = 0.18,
    ang_vel_threshold: float = 0.35,
    upright_threshold: float = -0.85,
    leg_joint_names: list[str] | None = None,
    l1: float = 0.21665632675675972,
    l2: float = 0.2540023491164531,
    offset: float = -0.007712217793726145,
    theta1_offset: float = 0.14299916248023697,
    theta2_offset: float = 2.406020345452543,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse stand success: upright, wheels below body, tall enough, and quiet."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_length, theta0 = _recovery_stand_leg_state(
        env, leg_joint_names, l1, l2, offset, theta1_offset, theta2_offset, asset_cfg
    )

    upright = asset.data.projected_gravity_b[:, 2] < upright_threshold
    tall = asset.data.root_pos_w[:, 2] > (target_height - height_margin)
    long_enough = torch.all(leg_length > min_leg_length, dim=1)
    wheels_under_body = torch.all(torch.abs(theta0) < theta0_threshold, dim=1)
    quiet_lin = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < lin_vel_threshold
    quiet_ang = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1) < ang_vel_threshold
    return (upright & tall & long_enough & wheels_under_body & quiet_lin & quiet_ang).float()


def recovery_stand_wheel_vel_l2(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.85,
    fallen_threshold: float = -0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sustained wheel spin once upright; wheels may still assist recovery."""
    asset: Articulation = env.scene[asset_cfg.name]
    phase = recovery_stand_upright_factor(env, upright_threshold, fallen_threshold, asset_cfg)
    return phase * torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def recovery_wheel_assist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward using wheels to assist in stand-up recovery.

    When wheels contact the ground, driving them can help rotate the body
    toward upright. This reward encourages wheel-ground contact during recovery.

    Args:
        env: The RL environment.
        asset_cfg: The asset configuration.

    Returns:
        Reward value, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward having at least some angular velocity when not upright
    # This encourages active motion rather than staying still while fallen
    ang_vel_magnitude = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    # More reward for active motion when clearly fallen
    fallen_mask = is_fallen(env).float()
    reward = ang_vel_magnitude * fallen_mask
    # Clamp to prevent excessive spinning
    reward = torch.clamp(reward, max=5.0)
    return reward


def recovery_mode_flag(
    env: ManagerBasedRLEnv,
    upright_threshold: float = -0.7,
    fallen_threshold: float = -0.3,
) -> torch.Tensor:
    """Recovery mode flag for observation: 0 = normal locomotion, 1 = recovery.

    Smoothly interpolates between 0 (upright) and 1 (fallen) based on
    projected_gravity_z. This gives the policy a continuous signal about
    its orientation state.

    Args:
        env: The RL environment.
        upright_threshold: Below this proj_gravity_z, robot is fully "upright" (flag=0).
        fallen_threshold: Above this proj_gravity_z, robot is fully "fallen" (flag=1).

    Returns:
        Recovery mode flag, shape (num_envs, 1). Range [0, 1].
    """
    proj_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    # Linear interpolation between upright_threshold (0) and fallen_threshold (1)
    flag = torch.clamp((proj_z - upright_threshold) / (fallen_threshold - upright_threshold), min=0.0, max=1.0)
    return flag.unsqueeze(-1)


def upright_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute the upright scaling factor used to modulate locomotion rewards.

    Same as the inline clamp used throughout existing rewards, but exposed
    as a standalone function for reuse in blended reward computation.

    Returns:
        Scaling factor in [0, 1], shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
