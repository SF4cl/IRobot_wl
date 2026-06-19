"""PPO runner configuration for VMC flat recovery (stand-up + locomotion) training.

Warm-starts from the best flat locomotion checkpoint and continues training
with recovery-specific environment configuration.
"""

from isaaclab.utils import configclass

from .vmc_rsl_rl_ppo_cfg import WLVMCVanillaFlatPPORunnerCfg


@configclass
class WLVMCVanillaFlatRecoveryPPORunnerCfg(WLVMCVanillaFlatPPORunnerCfg):
    """PPO runner config for VMC flat + stand-up recovery training.

    Inherits the network architecture and hyperparameters from the standard
    flat VMC config. Key differences:
      - More iterations (8000 vs 5000) for the harder multi-skill task
      - Different experiment name for log organization
      - Designed to warm-start from a pre-trained flat locomotion checkpoint
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 8000
        self.experiment_name = "wl_vmc_flat_recovery"

        # Warm-start: path to a pre-trained checkpoint (set before training)
        # Example: "logs/rsl_rl/wl_vmc_flat/2026-06-19_01-45-54/model_5000.pt"
        self.load_checkpoint = None
        self.load_run = None

        # Save more frequently to track recovery skill acquisition
        self.save_interval = 50

        # Same network architecture as flat (27-dim policy obs, 5-step history)
        # policy.num_encoder_obs = 27 * 5 = 135  (inherited)
        # policy.latent_dim = 3                    (inherited)
        # policy.encoder_hidden_dims = [128, 64]   (inherited)
        # policy.actor_hidden_dims = [128, 64, 32] (inherited)
        # policy.critic_hidden_dims = [256, 128, 64] (inherited)

        # Slightly lower learning rate for fine-tuning with new task
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.extra_learning_rate = 5.0e-4
