import gymnasium as gym

from . import agents, go1_flat_env_cfg, go1_rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Go1-Flat-v0",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_flat_env_cfg.Go1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Go1-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_flat_env_cfg.Go1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Go1-Rough-v0",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_rough_env_cfg.Go1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go1RoughPPORunnerCfg,
    },
)

gym.register(
    id="Go1-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_rough_env_cfg.Go1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go1RoughPPORunnerCfg,
    },
)