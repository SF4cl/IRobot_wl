# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .vmc_rough_env_cfg import WLVMCVanillaRoughEnvCfg


STAGE_B1_GLIDE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.50,
        ),
        "micro_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.03, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
        # Stage B1 keeps only low 3 cm curbs so the policy first learns
        # to absorb clear wheel-edge disturbances without immediately
        # overfitting to taller step obstacles.
        "curb_3cm": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.10,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=8,
                height=0.03,
                size=(0.05, 0.15),
                max_yx_angle=0.0,
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=10,
                height=0.03,
                size=(0.05, 0.15),
                max_yx_angle=0.0,
            ),
        ),
    },
)


@configclass
class WLVMCVanillaFlatEnvCfg(WLVMCVanillaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spread out environments for better visualization
        self.scene.env_spacing = 4.0

        # override rewards
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_enhance.weight = 1.2
        self.rewards.base_height_enhance.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        # Stage B1: reduce pure flat terrain, make the micro-rough surface
        # noticeably harsher, and add only low 3 cm curbs as an
        # introductory edge-obstacle curriculum.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAGE_B1_GLIDE_TERRAIN_CFG
        # Keep terrain sensing for critic / base-height shaping, but leave
        # policy observation compact like the old WL-Gym VMC setup.
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLVMCVanillaFlatEnvCfg":
            self.disable_zero_weight_rewards()
