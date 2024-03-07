import gymnasium as gym

#from hcrl_orbit.locomotion.velocity.config.draco import agents, flat_env_cfg, rough_env_cfg
from . import agents, flat_env_cfg, rough_env_cfg, wbc_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="HCRL-Velocity-Draco-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DracoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoFlatPPORunnerCfg,
        },
)

gym.register(
    id="HCRL-Velocity-Draco-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DracoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoFlatPPORunnerCfg,
        },
)

gym.register(
    id="HCRL-Velocity-Rough-Draco-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.DracoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoRoughPPORunnerCfg,
        },
)

gym.register(
    id="HCRL-Velocity-Rough-Draco-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.DracoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoRoughPPORunnerCfg,
        },
)

gym.register(
    id="HCRL-WBC-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wbc_env_cfg.WBCEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoWBCPPORunnerCfg,
        },
)

gym.register(
    id="HCRL-WBC-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wbc_env_cfg.WBCEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DracoWBCPPORunnerCfg,
        },
)