import gymnasium as gym

#from hcrl_orbit.locomotion.velocity.config.draco import agents, flat_env_cfg, rough_env_cfg
from . import agents, flat_env_cfg, PIC_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Bumpybot-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BumpybotFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)

gym.register(
    id="Bumpybot-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BumpybotFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)

gym.register(
    id="Bumpybot-PIC-Unicycle-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PIC_env_cfg.UnicycleEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)

gym.register(
    id="Bumpybot-PIC-Integrator-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PIC_env_cfg.IntegratorEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)

gym.register(
    id="Bumpybot-PIC-Car-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PIC_env_cfg.CarEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BumpybotFlatPPORunnerCfg,
        },
)