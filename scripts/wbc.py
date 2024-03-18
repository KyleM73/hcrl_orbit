"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting.")
parser.add_argument("--onnx", action="store_true", default=False, help="Convert the .pt file to a .onnx file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

# launch omniverse app
app_launcher = AppLauncher(args_cli, experience=app_experience)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import traceback
from datetime import datetime

import carb
from rsl_rl.runners import OnPolicyRunner

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import orbit.hcrl  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
import omni.isaac.orbit.utils.math as math_utils

if args_cli.plot:
    import matplotlib.pyplot as plt

from orbit.hcrl.tasks.locomotion.mdp.pnc.config.draco3_config import SimConfig
from orbit.hcrl.tasks.locomotion.mdp.pnc.draco3_pnc.draco3_interface import Draco3Interface
from orbit.hcrl.tasks.locomotion.mdp.pnc.util import util
from orbit.hcrl.tasks.locomotion.mdp.pnc.util import liegroup

def format_obs(asset, obs_dict):
    del obs_dict["policy"]
    joint_idx, joint_names = asset.find_joints(name_keys=[".*"])
    for k,v in obs_dict.items():
        if "contact" in k:
            obs_dict[k] = bool(v.cpu().numpy()) #need to make array for batched control
        elif "quat" in k:
            obs_dict[k] = math_utils.convert_quat(v, "xyzw").view(-1).cpu().numpy()
        elif k in ["joint_pos", "joint_vel"]:
            obs_dict[k] = {}
            for i in range(len(joint_idx)):
                obs_dict[k][joint_names[i]] = v.view(-1)[joint_idx[i]].cpu().numpy() 
        else:
            obs_dict[k] = v.cpu().view(-1).numpy()
    return obs_dict

def main():
    """Train with RSL-RL agent."""
    # parse configuration
    print("0")
    env_cfg: RLTaskEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    print("1")
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print("2")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        #env.metadata["video.frames_per_second"] = 1.0 / env.unwrapped.step_dt
    # wrap around environment for rsl-rl
    print("3")
    env = RslRlVecEnvWrapper(env)
    print("4")
    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # reset environment
    print("5")
    obs, extra = env.get_observations()
    obs_dict = format_obs(env.unwrapped.scene["robot"], extra["observations"])
    step = 0
    # Construct Interface
    print("constructing interface...")
    interface = Draco3Interface()
    print("done.")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            #actions = policy(obs) #pypnc.interface.get_command(data)
            command = interface.get_command(obs_dict)
            print(command)
            # env stepping
            obs, _, _, extras = env.step(command)
            obs_dict = format_obs(env.unwrapped.scene["robot"], extra["observations"])
            #print(obs)
            #print()
            if args_cli.video: env.unwrapped.render()
        if not step % 50: print("Step {}/{}...".format(step, args_cli.video_length))
        if step > args_cli.video_length: break
        if env.unwrapped.sim.is_stopped(): break
        step += 1

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
