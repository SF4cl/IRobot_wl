# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
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

        # Continuous reference-style velocity, heading, and height commands.
        self.commands.base_velocity.class_type = mdp.ContinuousHeightVelocityCommand
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.base_height_range = (0.19, 0.28)
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.14, 3.14)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.5
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.observations.policy.velocity_commands.params["height_command"] = 0.235
        self.observations.critic.velocity_commands.params["height_command"] = 0.235

        # Reference Wheel-Legged-Gym reward set.
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -10.0

        self.rewards.base_height_l2.weight = 1.0
        self.rewards.base_height_l2.func = mdp.ref_command_base_height_exp
        self.rewards.base_height_l2.params["command_name"] = "base_velocity"
        self.rewards.base_height_l2.params["fallback_target_height"] = 0.235
        self.rewards.base_height_l2.params["asset_cfg"] = SceneEntityCfg("robot")
        self.rewards.base_height_l2.params.pop("sensor_cfg", None)
        self.rewards.base_height_l2.params.pop("target_height", None)
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

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*f0_Link",
            ".*f1_Link",
        ]
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.contact_forces.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.func = mdp.ref_track_lin_vel_exp
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "tracking_sigma": 0.25}
        self.rewards.tracking_lin_vel_enhance.func = mdp.ref_track_lin_vel_enhance
        self.rewards.tracking_lin_vel_enhance.weight = 1.0
        self.rewards.tracking_lin_vel_enhance.params = {"command_name": "base_velocity", "tracking_sigma": 0.25}
        self.rewards.track_ang_vel_z_exp.func = mdp.ref_track_ang_vel_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params = {"command_name": "base_velocity", "tracking_sigma": 0.25}

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

        # Resume the command curriculum at +/-1.6 m/s and expand to +/-2.5 m/s.
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.64, 1.0)
        self.curriculum.command_levels_lin_vel.params["threshold"] = 0.42
        self.curriculum.command_levels_lin_vel.params["step_size"] = 0.1
        self.curriculum.command_levels_lin_vel.params["update_interval_s"] = 100.0
        self.curriculum.command_levels_ang_vel = None

        # Reference-like reset robustness, without an extra reset impulse.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.1, 2.0)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_reset_base.params["pose_range"] = {}
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }

        # Leg-link contact is penalized above; only sustained base contact or severe tilt terminates.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.sustained_flat_failure,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]),
                "contact_threshold": 10.0,
                "projected_gravity_z_threshold": -0.1,
                "hold_time_s": 1.0,
            },
        )
        self.terminations.terrain_out_of_bounds = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaFlatEnvCfg":
            self.disable_zero_weight_rewards()
