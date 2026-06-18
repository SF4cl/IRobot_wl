# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

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
        self.episode_length_s = 60.0

        # override rewards
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_enhance.weight = 0.6
        self.rewards.base_height_enhance.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Override default leg length to be shorter for flat terrain
        self.vmc_actions.l0_offset = 0.17
        self.actions.vmc.l0_offset = 0.17
        self.vmc_actions.action_scale_vel = 52.0
        self.actions.vmc.action_scale_vel = 52.0
        self.vmc_actions.wheel_damping = 0.08
        self.actions.vmc.wheel_damping = 0.08

        # Disable feet_distance_y_exp: wheel Y is constant (kinematic invariant),
        # the policy cannot control it through actions.
        self.rewards.feet_distance_y_exp.weight = 0.0

        self.rewards.vmc_action_symmetry.weight = -0.1
        self.rewards.vmc_action_symmetry.params["wheel_scale"] = 0.0
        self.rewards.nominal_state.weight = -0.05
        self.rewards.leg_length_symmetry.weight = -1.0
        self.rewards.leg_angle_symmetry.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_pos_penalty.weight = -0.1
        self.rewards.stand_still.weight = -0.5
        self.rewards.stand_still.params["lin_x_threshold"] = 0.05
        self.rewards.stand_still.params["ang_z_threshold"] = 0.05
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.wheel_vel_penalty.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.tracking_lin_vel_enhance.weight = 0.0
        self.rewards.tracking_ang_vel_enhance.weight = 0.0

        self.commands.base_velocity.class_type = mdp.MixedModeVelocityCommand
        self.commands.base_velocity.mode_probabilities = (0.1, 0.3, 0.3, 0.3)
        self.commands.base_velocity.min_lin_vel_x = 0.15
        self.commands.base_velocity.min_ang_vel_z = 0.25
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.3, 1.0)
        self.curriculum.command_levels_lin_vel.params["threshold"] = 0.4
        self.curriculum.command_levels_lin_vel.params["step_size"] = 0.1
        self.curriculum.command_levels_lin_vel.params["update_interval_s"] = 20.0
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.4, 1.0)
        self.curriculum.command_levels_ang_vel.params["threshold"] = 0.4
        self.curriculum.command_levels_ang_vel.params["step_size"] = 0.1
        self.curriculum.command_levels_ang_vel.params["update_interval_s"] = 20.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaFlatEnvCfg":
            self.disable_zero_weight_rewards()
