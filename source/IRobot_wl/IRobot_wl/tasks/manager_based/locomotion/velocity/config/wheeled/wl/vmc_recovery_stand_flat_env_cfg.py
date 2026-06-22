"""Self-righting plus in-place stand-up task for the WL robot."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_recovery_flat_env_cfg import WLVMCRecoveryFlatEnvCfg


@configclass
class WLVMCRecoveryStandFlatEnvCfg(WLVMCRecoveryFlatEnvCfg):
    """Recover from fallen poses, then stand up in place."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.low_stand_base_height = 0.19
        self.low_stand_leg_length = 0.15

        # Allow a small amount of wheel authority for balance and in-place
        # correction. Keep this far below locomotion torque during this stage.
        self.vmc_actions.action_scale_wheel_torque = 0.75
        self.vmc_actions.clip_wheel_actions = 1.0
        self.actions.vmc.action_scale_wheel_torque = self.vmc_actions.action_scale_wheel_torque
        self.actions.vmc.clip_wheel_actions = self.vmc_actions.clip_wheel_actions

        # Slightly reduce the standing feedforward so the policy must learn
        # active extension instead of simply jamming both legs into the ground.
        self.vmc_actions.feedforward_force = 32.0
        self.actions.vmc.feedforward_force = self.vmc_actions.feedforward_force

        # Start with mostly fallen resets, but include some near-upright states
        # so the policy sees the stand-up phase frequently.
        self.events.randomize_reset_base.params["pose_range"]["roll"] = (-1.25, 1.25)
        self.events.randomize_reset_base.params["pose_range"]["pitch"] = (2.7, 3.14)
        self.events.randomize_reset_joints.params["position_range"] = (-0.25, 0.25)

        # Keep the stand-up stage on the lowest fixed target first.  The flat
        # locomotion task randomizes this command, but RecoveryStand should
        # learn a stable low stand before target-height variation is added.
        self.commands.base_velocity.base_height_range = (
            self.low_stand_base_height,
            self.low_stand_base_height,
        )
        self.observations.policy.velocity_commands.params["height_command"] = self.low_stand_base_height
        self.observations.critic.velocity_commands.params["height_command"] = self.low_stand_base_height

        self._disable_all_rewards()

        # Recovery phase: keep the successful flip-up behavior.
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-2.0)
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.self_right_orientation_l2,
            weight=-6.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=mdp.self_right_lin_vel_z_l2,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=mdp.self_right_ang_vel_xy_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Stand-up phase: retract when fallen, extend after upright, place the
        # wheels under the body, and avoid walking away while standing up.
        self.rewards.recovery_stand_leg_length = RewTerm(
            func=mdp.recovery_stand_leg_length_l2,
            weight=-8.0,
            params={
                "retracted_length": 0.145,
                "standing_length": self.low_stand_leg_length,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_theta0 = RewTerm(
            func=mdp.recovery_stand_theta0_l2,
            weight=-1.5,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "leg_joint_names": self.leg_joint_names,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_stand_base_height = RewTerm(
            func=mdp.recovery_stand_base_height_l2,
            weight=-20.0,
            params={
                "target_height": self.low_stand_base_height,
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
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
        self.rewards.recovery_stand_ang_vel_z = RewTerm(
            func=mdp.recovery_stand_ang_vel_z_l2,
            weight=-0.2,
            params={
                "upright_threshold": -0.85,
                "fallen_threshold": -0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Regularization. Penalize leg effort more than wheel effort so the
        # wheels can help, but do not let them become the main solution.
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

        if self.__class__.__name__ == "WLVMCRecoveryStandFlatEnvCfg":
            self.disable_zero_weight_rewards()
