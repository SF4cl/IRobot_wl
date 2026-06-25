"""In-place stand-up task from upright body poses with randomized legs."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_flat_env_cfg import WLVMCVanillaFlatEnvCfg


@configclass
class WLVMCStandFlatEnvCfg(WLVMCVanillaFlatEnvCfg):
    """Stand in place from an upright body and randomized leg configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0
        self.low_stand_base_height = 0.19
        self.low_stand_leg_length = 0.18

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.policy.enable_corruption = False
        self.observations.policy_history.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # ------------------------------Actions------------------------------
        self.vmc_actions.feedforward_force = 0.0
        self.vmc_actions.action_scale_tp = 50.0
        self.vmc_actions.action_scale_force = 90.0
        self.vmc_actions.action_scale_wheel_torque = 0.75
        self.vmc_actions.clip_tp_actions = 1.0
        self.vmc_actions.clip_force_actions = 1.0
        self.vmc_actions.clip_wheel_actions = 1.0
        self.vmc_actions.torque_limits = [50.0, 50.0, 50.0, 50.0, 4.0, 4.0]

        self.actions.vmc.feedforward_force = self.vmc_actions.feedforward_force
        self.actions.vmc.action_scale_tp = self.vmc_actions.action_scale_tp
        self.actions.vmc.action_scale_force = self.vmc_actions.action_scale_force
        self.actions.vmc.action_scale_wheel_torque = self.vmc_actions.action_scale_wheel_torque
        self.actions.vmc.clip_tp_actions = self.vmc_actions.clip_tp_actions
        self.actions.vmc.clip_force_actions = self.vmc_actions.clip_force_actions
        self.actions.vmc.clip_wheel_actions = self.vmc_actions.clip_wheel_actions
        self.actions.vmc.torque_limits = self.vmc_actions.torque_limits
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

        # Upright body, random yaw, low-ish base height, randomized legs.
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform
        self.events.randomize_reset_base.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.06, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
            "position_range": (-0.8, 0.8),
            "velocity_range": (0.0, 0.0),
        }

        # Zero command term for observation compatibility.
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.base_height_range = (
            self.low_stand_base_height,
            self.low_stand_base_height,
        )
        self.observations.policy.velocity_commands.params["height_command"] = self.low_stand_base_height
        self.observations.critic.velocity_commands.params["height_command"] = self.low_stand_base_height

        # ------------------------------Rewards------------------------------
        self._disable_all_rewards()

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.2)
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.self_right_orientation_l2,
            weight=-8.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_stand_theta0 = RewTerm(
            func=mdp.recovery_stand_theta0_l2,
            weight=-45.0,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_theta0_exp = RewTerm(
            func=mdp.recovery_stand_theta0_exp,
            weight=6.0,
            params={
                "std": 0.45,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_theta0_worst = RewTerm(
            func=mdp.recovery_stand_theta0_worst_l2,
            weight=-15.0,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_splayed_long_leg = RewTerm(
            func=mdp.recovery_stand_splayed_long_leg_l2,
            weight=-20.0,
            params={
                "theta0_threshold": 0.8,
                "leg_length_threshold": self.low_stand_leg_length,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_leg_length = RewTerm(
            func=mdp.recovery_stand_leg_length_l2,
            weight=-30.0,
            params={
                "retracted_length": self.low_stand_leg_length,
                "standing_length": self.low_stand_leg_length,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_leg_symmetry = RewTerm(
            func=mdp.recovery_stand_leg_symmetry_l2,
            weight=-3.0,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_base_height = RewTerm(
            func=mdp.recovery_stand_base_height_l2,
            weight=-110.0,
            params={
                "target_height": self.low_stand_base_height,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "gate_by_theta0": True,
                "theta0_ready_std": 0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_wheel_load = RewTerm(
            func=mdp.recovery_stand_wheel_load,
            weight=5.0,
            params={
                "target_total_force_n": 100.0,
                "min_each_force_n": 35.0,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["l_wheel_Link", "r_wheel_Link"],
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_wheel_load_deficit = RewTerm(
            func=mdp.recovery_stand_wheel_load_deficit,
            weight=-6.0,
            params={
                "target_total_force_n": 100.0,
                "min_each_force_n": 35.0,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["l_wheel_Link", "r_wheel_Link"],
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_success_bonus = RewTerm(
            func=mdp.recovery_stand_success_bonus,
            weight=8.0,
            params={
                "target_height": self.low_stand_base_height,
                "min_leg_length": 0.17,
                "theta0_threshold": 0.35,
                "height_margin": 0.015,
                "lin_vel_threshold": 0.18,
                "ang_vel_threshold": 0.35,
                "upright_threshold": -0.85,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_lin_vel_xy = RewTerm(
            func=mdp.recovery_stand_lin_vel_xy_l2,
            weight=-0.7,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=mdp.self_right_ang_vel_xy_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_stand_ang_vel_z = RewTerm(
            func=mdp.recovery_stand_ang_vel_z_l2,
            weight=-0.2,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_negative_force = RewTerm(
            func=mdp.recovery_stand_negative_force_l2,
            weight=-0.3,
            params={
                "action_name": "vmc",
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_leg_ground_contact = RewTerm(
            func=mdp.recovery_stand_leg_ground_contact,
            weight=-0.5,
            params={
                "threshold": 1.0,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["lf0_Link", "rf0_Link", "lf1_Link", "rf1_Link"],
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.joint_vel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0015,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.joint_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-8.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.recovery_stand_wheel_vel = RewTerm(
            func=mdp.recovery_stand_wheel_vel_l2,
            weight=-0.003,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
            },
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.008)

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.stand_not_upright = DoneTerm(
            func=mdp.recovery_stand_timeout_not_upright,
            params={
                "max_time_s": 1.5,
                "min_upright_factor": 0.65,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.terminations.stand_not_standing = DoneTerm(
            func=mdp.recovery_stand_timeout_not_standing,
            params={
                "max_time_s": 3.5,
                "min_upright_factor": 0.85,
                "min_height": 0.185,
                "target_total_force_n": 100.0,
                "min_each_force_n": 35.0,
                "min_load_score": 0.65,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["l_wheel_Link", "r_wheel_Link"],
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "WLVMCStandFlatEnvCfg":
            self.disable_zero_weight_rewards()

    def _disable_all_rewards(self):
        """Clear inherited locomotion rewards before adding stand terms."""
        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            reward_attr = getattr(self.rewards, attr)
            if reward_attr is None or callable(reward_attr):
                continue
            if hasattr(reward_attr, "weight"):
                setattr(self.rewards, attr, None)
