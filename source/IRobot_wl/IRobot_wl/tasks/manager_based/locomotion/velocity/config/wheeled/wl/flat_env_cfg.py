from isaaclab.utils import configclass

from .rough_env_cfg import WLRoughEnvCfg


@configclass
class WLFlatEnvCfg(WLRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # spread out environments for better visualization
        self.scene.env_spacing = 4.0

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # Use a single infinite plane so envs are separated by env_spacing
        # instead of all being assigned to the same 1x1 terrain patch.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WLFlatEnvCfg":
            self.disable_zero_weight_rewards()
