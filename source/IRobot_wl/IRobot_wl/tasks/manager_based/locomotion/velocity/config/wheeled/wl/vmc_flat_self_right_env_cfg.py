"""VMC flat self-righting environment.

This environment isolates the first recovery stage:
fallen body attitude -> projected gravity close to [0, 0, -1].
Wheels are passive and locomotion commands/rewards are disabled.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_flat_env_cfg import WLVMCVanillaFlatEnvCfg


@configclass
class WLVMCVanillaFlatSelfRightEnvCfg(WLVMCVanillaFlatEnvCfg):
    """VMC-only self-righting task with passive wheels."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # ------------------------------Actions------------------------------
        # Keep the 6D VMC action layout for runner compatibility, but make the
        # wheel channels produce exactly zero torque.
        self.vmc_actions.action_scale_vel = 0.0
        self.actions.vmc.action_scale_vel = 0.0
        self.vmc_actions.clip_wheel_actions = 0.0
        self.actions.vmc.clip_wheel_actions = 0.0
        self.vmc_actions.wheel_damping = 0.0
        self.actions.vmc.wheel_damping = 0.0

        # Slightly reduce delay/randomization for the first isolated skill.
        self.actions.vmc.randomize_action_delay = False

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_root_state_fallen,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names),
                "pose_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                    "roll": (-3.14, 3.14),
                    "pitch": (-3.14, 3.14),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
                "joint_position_range": (-0.8, 0.8),
                "joint_velocity_range": (-0.5, 0.5),
                "fallen_probability": 1.0,
                "ground_height_offset": 0.06,
            },
        )
        self.events.randomize_reset_joints = None
        self.events.randomize_push_robot = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        self.commands.base_velocity.min_lin_vel_x = 0.0
        self.commands.base_velocity.min_ang_vel_z = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------Terminations------------------------------
        # Ground contact by the body and links is expected while self-righting.
        self.terminations.illegal_contact = None

        # ------------------------------Rewards------------------------------
        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            reward_attr = getattr(self.rewards, attr)
            if hasattr(reward_attr, "weight"):
                reward_attr.weight = 0.0

        self.rewards.self_right_attitude = RewTerm(
            func=mdp.self_right_attitude,
            weight=6.0,
            params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.self_right_tilt_progress = RewTerm(
            func=mdp.self_right_tilt_progress,
            weight=25.0,
            params={"max_reward": 0.05, "max_penalty": 0.02, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.self_right_upright_success = RewTerm(
            func=mdp.self_right_upright_success,
            weight=2.0,
            params={"threshold": -0.85, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.ang_vel_xy_l2.weight = -0.03
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_torques_l2.weight = -1.0e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = ["lf1_Joint", "rf1_Joint"]

        # ------------------------------Curriculum------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "WLVMCVanillaFlatSelfRightEnvCfg":
            self.disable_zero_weight_rewards()
