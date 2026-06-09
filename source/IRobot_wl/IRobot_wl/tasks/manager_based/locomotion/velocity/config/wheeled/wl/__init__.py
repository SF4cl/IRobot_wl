import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IRobot-WL-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WLFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WLFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:WLFlatTrainerCfg",
    },
)

gym.register(
    id="IRobot-WL-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WLRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WLRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:WLRoughTrainerCfg",
    },
)

##
# VMC (Virtual Model Control) environments.
##

gym.register(
    id="IRobot-WL-Velocity-VMC-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vmc_flat_env_cfg:WLVMCVanillaFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.vmc_rsl_rl_ppo_cfg:WLVMCVanillaFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.vmc_cusrl_ppo_cfg:WLVMCVanillaFlatTrainerCfg",
    },
)

gym.register(
    id="IRobot-WL-Velocity-VMC-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vmc_rough_env_cfg:WLVMCVanillaRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.vmc_rsl_rl_ppo_cfg:WLVMCVanillaRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.vmc_cusrl_ppo_cfg:WLVMCVanillaRoughTrainerCfg",
    },
)
