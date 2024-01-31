import gymnasium as gym

#from hcrl_orbit.locomotion.velocity.config.draco import agents, flat_env_cfg, rough_env_cfg
from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-HCRL-Navigation-Bumpybot-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BumpybotFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)

gym.register(
    id="Isaac-HCRL-Navigation-Bumpybot-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BumpybotFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)