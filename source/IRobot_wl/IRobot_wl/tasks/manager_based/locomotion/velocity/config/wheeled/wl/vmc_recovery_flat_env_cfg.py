"""Reference-style pure self-righting task for the WL robot.

This task intentionally keeps only the pieces needed to learn stand-up from
fallen poses.  It reuses the existing VMC action/observation plumbing so the
policy interface remains compatible with the WL sequence runner, but removes
locomotion tracking, push/randomization curriculum, contact termination, and
wheel drive authority.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_flat_env_cfg import WLVMCVanillaFlatEnvCfg


@configclass
class WLVMCRecoveryFlatEnvCfg(WLVMCVanillaFlatEnvCfg):
    """Flat-ground pure self-righting task."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # Recovery is easier to debug without sensor noise and domain
        # randomization.  Add these back only after the basic behavior is stable.
        self.observations.policy.enable_corruption = False
        self.observations.policy_history.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # ------------------------------Actions------------------------------
        # Keep the 6-D VMC action interface but make this a leg-only recovery
        # task.  The previous recovery run let the policy spin the wheels into
        # extreme velocities, eventually corrupting the simulation with NaNs.
        self.vmc_actions.l0_offset = 0.13
        self.vmc_actions.feedforward_force = 0.0
        self.vmc_actions.action_scale_tp = 15.0
        self.vmc_actions.action_scale_force = 25.0
        self.vmc_actions.action_scale_wheel_torque = 0.0
        self.vmc_actions.clip_tp_actions = 1.0
        self.vmc_actions.clip_force_actions = 1.0
        self.vmc_actions.clip_wheel_actions = 0.0

        self.actions.vmc.l0_offset = self.vmc_actions.l0_offset
        self.actions.vmc.feedforward_force = self.vmc_actions.feedforward_force
        self.actions.vmc.action_scale_tp = self.vmc_actions.action_scale_tp
        self.actions.vmc.action_scale_force = self.vmc_actions.action_scale_force
        self.actions.vmc.action_scale_wheel_torque = self.vmc_actions.action_scale_wheel_torque
        self.actions.vmc.clip_tp_actions = self.vmc_actions.clip_tp_actions
        self.actions.vmc.clip_force_actions = self.vmc_actions.clip_force_actions
        self.actions.vmc.clip_wheel_actions = self.vmc_actions.clip_wheel_actions
        self.actions.vmc.randomize_vmc_gains = False
        self.actions.vmc.randomize_action_delay = False
        self.actions.vmc.action_delay_ms_range = (0.0, 0.0)

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # Match the reference recovery reset: low base, inverted/side-biased
        # orientation distribution, zero root velocity.
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform
        self.events.randomize_reset_base.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                # reset_root_state_uniform adds this to the default root height
                "z": (-0.12, -0.12),
                "roll": (-0.5 * math.pi, 0.5 * math.pi),
                "pitch": (math.pi, math.pi),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        }

        # Keep a zero command term for observation compatibility only.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0

        # ------------------------------Rewards------------------------------
        self._disable_all_rewards()

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-2.0)
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.self_right_orientation_l2,
            weight=-8.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=mdp.self_right_lin_vel_z_l2,
            weight=-0.03,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_xy_l2 = RewTerm(
            func=mdp.self_right_lin_vel_xy_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=mdp.self_right_ang_vel_xy_l2,
            weight=-0.08,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.self_right_upright_bonus = RewTerm(
            func=mdp.self_right_upright_bonus,
            weight=2.0,
            params={
                "ang_vel_std": 0.8,
                "start_upright": 0.75,
                "full_upright": 0.9,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.002,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.upright_joint_vel_l2 = RewTerm(
            func=mdp.self_right_upright_joint_vel_l2,
            weight=-0.01,
            params={
                "start_upright": 0.75,
                "full_upright": 0.9,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
            },
        )
        self.rewards.joint_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        self.rewards.upright_action_rate_l2 = RewTerm(
            func=mdp.self_right_upright_action_rate_l2,
            weight=-0.03,
            params={
                "start_upright": 0.75,
                "full_upright": 0.9,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "WLVMCRecoveryFlatEnvCfg":
            self.disable_zero_weight_rewards()

    def _disable_all_rewards(self):
        """Clear inherited locomotion rewards before adding recovery terms."""
        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            reward_attr = getattr(self.rewards, attr)
            if reward_attr is None or callable(reward_attr):
                continue
            if hasattr(reward_attr, "weight"):
                setattr(self.rewards, attr, None)
