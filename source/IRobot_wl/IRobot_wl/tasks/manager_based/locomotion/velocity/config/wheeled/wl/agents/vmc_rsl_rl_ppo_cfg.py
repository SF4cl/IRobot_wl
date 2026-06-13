from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WLVMCVanillaRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "WlSequenceRunner"
    num_steps_per_env = 48
    max_iterations = 5000
    save_interval = 100
    experiment_name = "wl_vmc_rough"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor_obs_group = "policy"
    critic_obs_group = "critic"
    obs_history_group = "policy_history"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    policy.num_encoder_obs = 27 * 5
    policy.latent_dim = 3
    policy.encoder_hidden_dims = [128, 64]
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
    algorithm.extra_learning_rate = 1.0e-3


@configclass
class WLVMCVanillaFlatPPORunnerCfg(WLVMCVanillaRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "wl_vmc_flat"
