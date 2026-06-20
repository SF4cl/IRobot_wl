"""PPO runner configuration for VMC flat self-righting training."""

from isaaclab.utils import configclass

from .vmc_rsl_rl_ppo_cfg import WLVMCVanillaFlatPPORunnerCfg


@configclass
class WLVMCVanillaFlatSelfRightPPORunnerCfg(WLVMCVanillaFlatPPORunnerCfg):
    """Runner config for the isolated VMC self-righting task."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "wl_vmc_flat_self_right"
        self.save_interval = 50

        # The isolated task is shorter and easier to destabilize with excessive
        # exploration than locomotion.
        self.policy.init_noise_std = 0.35
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.extra_learning_rate = 5.0e-4
