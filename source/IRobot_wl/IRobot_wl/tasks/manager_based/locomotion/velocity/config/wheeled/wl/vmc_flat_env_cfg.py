# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .vmc_rough_env_cfg import WLVMCVanillaRoughEnvCfg


@configclass
class WLVMCVanillaFlatEnvCfg(WLVMCVanillaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spread out environments for better visualization
        self.scene.env_spacing = 4.0

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

        # Disable feet_distance_y_exp: wheel Y is constant (kinematic invariant),
        # the policy cannot control it through actions.
        self.rewards.feet_distance_y_exp.weight = 0.0

        self.rewards.vmc_action_symmetry.weight = 0.0
        self.rewards.nominal_state.weight = -0.1
        self.rewards.leg_length_symmetry.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.tracking_lin_vel_enhance.weight = 0.0
        self.rewards.tracking_ang_vel_enhance.weight = 0.0

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaFlatEnvCfg":
            self.disable_zero_weight_rewards()
