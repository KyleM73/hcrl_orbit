import gymnasium as gym

#from hcrl_orbit.locomotion.velocity.config.draco import agents, flat_env_cfg, rough_env_cfg
from . import agents, test_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="HCRL-Test-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": test_env_cfg.TestEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TestPPORunnerCfg,
        },
)