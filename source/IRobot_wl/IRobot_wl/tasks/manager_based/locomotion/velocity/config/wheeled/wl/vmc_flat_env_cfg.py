# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import IRobot_wl.tasks.manager_based.locomotion.velocity.mdp as mdp
from .vmc_rough_env_cfg import WLVMCVanillaRoughEnvCfg


@configclass
class WLVMCVanillaFlatEnvCfg(WLVMCVanillaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spread out environments for better visualization
        self.scene.env_spacing = 4.0
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # Continuous reference-style velocity, yaw-rate, and height commands.
        self.commands.base_velocity.class_type = mdp.ContinuousHeightVelocityCommand
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.base_height_range = (0.19, 0.28)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.heading_control_stiffness = 1.5
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.observations.policy.velocity_commands.func = mdp.wl_vmc_commands_with_recovery
        self.observations.critic.velocity_commands.func = mdp.wl_vmc_commands_with_recovery
        self.observations.policy.velocity_commands.params["height_command"] = 0.235
        self.observations.critic.velocity_commands.params["height_command"] = 0.235

        # Reference Wheel-Legged-Gym reward set.
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -4.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -10.0

        self.rewards.base_height_l2.weight = 3.0
        self.rewards.base_height_l2.func = mdp.recovery_low_to_command_base_height_exp
        self.rewards.base_height_l2.params = {
            "command_name": "base_velocity",
            "recovery_target_height": 0.155,
            "fallback_target_height": 0.235,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "non_wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]
            ),
            "leg_joint_names": self.leg_joint_names,
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.base_height_enhance.weight = 0.0

        self.rewards.joint_torques_l2.weight = -1.0e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0.0
        self.rewards.joint_vel_l2.weight = -5.0e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smooth = RewTerm(
            func=mdp.vmc_action_smooth_l2,
            weight=-0.01,
            params={"action_name": "vmc"},
        )

        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*f0_Link",
            ".*f1_Link",
        ]
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.contact_forces.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.func = mdp.recovery_gated_ref_track_lin_vel_exp
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "non_wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]
            ),
            "leg_joint_names": self.leg_joint_names,
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }
        self.rewards.tracking_lin_vel_enhance.func = mdp.recovery_gated_ref_track_lin_vel_enhance
        self.rewards.tracking_lin_vel_enhance.weight = 0.7
        self.rewards.tracking_lin_vel_enhance.params = {
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "non_wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]
            ),
            "leg_joint_names": self.leg_joint_names,
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }
        self.rewards.track_ang_vel_z_exp.func = mdp.recovery_gated_ref_track_ang_vel_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "base_velocity",
            "tracking_sigma": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "non_wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]
            ),
            "leg_joint_names": self.leg_joint_names,
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }

        # Reference nominal-state term: only left/right virtual-leg angle mismatch.
        self.rewards.leg_angle_symmetry.weight = -0.2
        self.rewards.theta0_nominal.weight = 0.0
        self.rewards.leg_length_symmetry.weight = 0.0
        self.rewards.vmc_action_symmetry.weight = 0.0

        # Disable additions that are not part of the reference reward set.
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.wheel_vel_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_sync.weight = 0.0
        self.rewards.applied_torque_limits.weight = 0.0
        self.rewards.ang_vel_z_cmd_l2.weight = 0.0
        self.rewards.track_lin_vel_enhance.weight = 0.0
        self.rewards.track_ang_vel_enhance.weight = 0.0
        self.rewards.tracking_ang_vel_enhance.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.upward.weight = 0.0
        self.rewards.self_right_orientation_l2.weight = -7.0
        self.rewards.recovery_upright_progress.weight = 4.0
        self.rewards.self_right_tilt_progress.weight = 30.0
        self.rewards.self_right_tilt_progress.params = {
            "max_reward": 0.06,
            "max_penalty": 0.01,
        }
        self.rewards.self_right_upright_success.weight = 2.0
        self.rewards.self_right_upright_success.params = {
            "threshold": -0.75,
        }
        self.rewards.recovery_stage_gate.weight = 0.0
        self.rewards.recovery_stand_min_leg_length.weight = -18.0
        self.rewards.recovery_stand_min_leg_length.params = {
            "target_length": self.vmc_actions.l0_min,
            "leg_joint_names": self.leg_joint_names,
            "wheel_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }
        self.rewards.recovery_stand_theta0.weight = -7.0
        self.rewards.recovery_stand_theta0.params = {
            "leg_joint_names": self.leg_joint_names,
            "wheel_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }
        self.rewards.recovery_stand_wheel_contact.weight = 2.0
        self.rewards.recovery_stand_wheel_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.recovery_stand_wheel_load.weight = 1.5
        self.rewards.recovery_stand_wheel_load.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.recovery_wheel_only_contact.weight = -4.0
        self.rewards.recovery_wheel_only_contact.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        self.rewards.recovery_still_lin_vel_xy.weight = -4.0
        self.rewards.recovery_still_lin_vel_xy.params = {"stage": 3}
        self.rewards.recovery_still_ang_vel_z.weight = -2.0
        self.rewards.recovery_still_ang_vel_z.params = {"stage": 3}
        self.rewards.recovery_still_wheel_vel.weight = -0.003
        self.rewards.recovery_still_wheel_vel.params = {
            "stage": 3,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.rewards.recovery_stage_wheel_vel.weight = -0.0008
        self.rewards.recovery_stage_wheel_vel.params = {
            "min_stage": 4,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.rewards.recovery_stage_wheel_torque.weight = -0.002
        self.rewards.recovery_stage_wheel_torque.params = {
            "min_stage": 4,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.rewards.recovery_stage_base_lin_vel_z.weight = -8.0
        self.rewards.recovery_stage_base_lin_vel_z.params = {
            "min_stage": 4,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_stage_leg_length_vel.weight = -0.25
        self.rewards.recovery_stage_leg_length_vel.params = {
            "min_stage": 4,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
            "leg_joint_names": self.leg_joint_names,
            "wheel_joint_names": self.wheel_joint_names,
            "l1": self.vmc_actions.l1,
            "l2": self.vmc_actions.l2,
            "offset": self.vmc_actions.offset,
            "theta1_offset": self.vmc_actions.theta1_offset,
            "theta2_offset": self.vmc_actions.theta2_offset,
        }
        self.rewards.recovery_stage_command_base_height_under.weight = -1200.0
        self.rewards.recovery_stage_command_base_height_under.params = {
            "command_name": "base_velocity",
            "min_stage": 4,
            "fallback_target_height": 0.235,
            "margin": 0.005,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_stage_command_base_height.weight = -800.0
        self.rewards.recovery_stage_command_base_height.params = {
            "command_name": "base_velocity",
            "min_stage": 4,
            "fallback_target_height": 0.235,
            "margin": 0.004,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_stage_force_action.weight = -0.01
        self.rewards.recovery_stage_force_action.params = {
            "action_name": "vmc",
            "min_stage": 4,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_stage_force_action_rate.weight = -0.08
        self.rewards.recovery_stage_force_action_rate.params = {
            "action_name": "vmc",
            "min_stage": 4,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_stage_force_action_symmetry.weight = -0.05
        self.rewards.recovery_stage_force_action_symmetry.params = {
            "action_name": "vmc",
            "min_stage": 4,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }
        self.rewards.recovery_zero_cmd_lin_vel_xy.weight = -3.0
        self.rewards.recovery_zero_cmd_lin_vel_xy.params = {
            "command_name": "base_velocity",
            "min_stage": 4,
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
        }
        self.rewards.recovery_zero_cmd_ang_vel_z.weight = -1.5
        self.rewards.recovery_zero_cmd_ang_vel_z.params = {
            "command_name": "base_velocity",
            "min_stage": 4,
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
        }
        self.rewards.recovery_zero_cmd_wheel_vel.weight = -0.002
        self.rewards.recovery_zero_cmd_wheel_vel.params = {
            "command_name": "base_velocity",
            "min_stage": 4,
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names),
        }
        self.rewards.recovery_zero_cmd_wheel_action.weight = -0.05
        self.rewards.recovery_zero_cmd_wheel_action.params = {
            "command_name": "base_velocity",
            "action_name": "vmc",
            "min_stage": 4,
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
        }
        self.rewards.recovery_zero_cmd_force_action.weight = -0.05
        self.rewards.recovery_zero_cmd_force_action.params = {
            "command_name": "base_velocity",
            "action_name": "vmc",
            "min_stage": 4,
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
            "start_time_s": 0.5,
            "ramp_time_s": 1.0,
        }

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Override default leg length to be shorter for flat terrain
        self.vmc_actions.l0_offset = 0.13
        self.actions.vmc.l0_offset = 0.13
        self.vmc_actions.feedforward_force = 0.0
        self.actions.vmc.feedforward_force = 0.0
        self.vmc_actions.action_scale_tp = 50.0
        self.actions.vmc.action_scale_tp = 50.0
        self.vmc_actions.action_scale_force = 100.0
        self.actions.vmc.action_scale_force = 100.0
        self.vmc_actions.action_scale_wheel_torque = 4.0
        self.actions.vmc.action_scale_wheel_torque = 4.0
        self.vmc_actions.clip_tp_actions = 1.0
        self.actions.vmc.clip_tp_actions = 1.0
        self.vmc_actions.clip_force_actions = 1.0
        self.actions.vmc.clip_force_actions = 1.0
        self.vmc_actions.clip_wheel_actions = 1.0
        self.actions.vmc.clip_wheel_actions = 1.0
        self.actions.vmc.zero_wheel_torque_until_upright = True
        self.actions.vmc.wheel_torque_upright_threshold = -0.72
        self.actions.vmc.wheel_torque_upright_ramp_width = 0.18

        self.curriculum.recovery_stages = CurrTerm(
            func=mdp.recovery_staged_curriculum,
            params={"stage_steps": (300, 900, 1500, 2100)},
        )

        # Keep locomotion commands tiny until the recovery curriculum reaches run stage.
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.0, 1.0)
        self.curriculum.command_levels_lin_vel.params["threshold"] = 0.42
        self.curriculum.command_levels_lin_vel.params["step_size"] = 0.1
        self.curriculum.command_levels_lin_vel.params["update_interval_s"] = 100.0
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_base_height = CurrTerm(
            func=mdp.command_levels_base_height,
            params={
                "initial_range": (0.19, 0.22),
                "final_range": (0.19, 0.28),
                "step_size": 0.02,
                "update_interval_iterations": 300,
            },
        )

        # Reference-like reset robustness, without an extra reset impulse.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.1, 2.0)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_reset_base.func = mdp.reset_root_state_fallen
        self.events.randomize_reset_base.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.15, 0.15),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.4, 0.4),
            },
            "joint_position_range": (-0.9, 0.9),
            "joint_velocity_range": (-0.8, 0.8),
            "fallen_probability": 1.0,
            "ground_height_offset": 0.14,
            "allow_random_orientation": True,
            "body_half_extents": (0.24, 0.17, 0.10),
            "spawn_height_margin": 0.06,
            "max_fallen_spawn_height": 0.42,
            "clamp_joint_positions": True,
            "reject_near_upright_fallen": True,
            "near_upright_projected_gravity_z": -0.55,
            "use_recovery_curriculum": True,
        }
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }

        # Recovery needs body/leg contacts during the early flip phase.
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaFlatEnvCfg":
            self.disable_zero_weight_rewards()
