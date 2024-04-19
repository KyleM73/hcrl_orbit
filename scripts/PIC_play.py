"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting.")

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

from omni.isaac.orbit.envs import BaseEnvCfg
from omni.isaac.orbit.utils.dict import print_dict
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.markers import VisualizationMarkersCfg, VisualizationMarkers

import orbit.hcrl  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

if args_cli.plot:
    import matplotlib.pyplot as plt

def PDcontroller(state, obs, device, P_vel=1, P_theta=1):
    pose_d, speed_d, theta_d = state[:, :2, 0], state[:, 2, 0], state[:, 3, 0]
    pose_diff = pose_d - obs[:, :2]
    theta = torch.arctan2(pose_diff[:, 0], pose_diff[:, 1])
    theta_err = theta - obs[:, 3]

    action = torch.tensor([P_vel * speed_d, P_theta * theta_err], device=device).view(-1, 2)
    return action

def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: BaseEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "PIC")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

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
        env.metadata["render_fps"] = 1.0 / env.unwrapped.step_dt
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # set seed of the environment
    env.seed(0)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # reset environment
    obs, _ = env.get_observations()
    step = 0

    # obtain the policy for inference
    obs[0, :2] = 4
    init_pose = obs[0, :2]
    pose = torch.zeros(1, 3, device=env.unwrapped.device)
    dt = 0.01
    T = args_cli.video_length * dt
    policy = orbit.hcrl.tasks.navigation.pic.PathIntegralController(obs, dt, T, num_samples=10_000, box_radius=0.5, device=env.unwrapped.device)

    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/testMarkers",
        markers={
            "sample": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            ),
            "true": sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    marker = VisualizationMarkers(cfg)
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            #state, actions, samples = policy(obs)
            state = torch.tensor([3.9, 3.9, 1, 3.14], device=env.unwrapped.device).view(1, 4, 1)
            action = PDcontroller(state, obs, env.unwrapped.device, 0, 1)
            print(action)
            

            #pose[0, :2] = state[0, :2, 0]# - init_pose
            #print(state[:, :2, 0])
            #print(init_pose)
            #print(obs[:, :2])
            #print(pose)
            #print()
            # visualize samples
            #pts = samples[::100, ::100, :3, 0] #sample every 10th point
            #pts[:, :, 2] = 0.2
            #pts = pts.flatten(0, 1)
            #x = state[:, :3, 0]
            #x[:, 2] = 0.3
            #pts = torch.cat((pts, x.view(-1, 3)), dim=0)
            #marker_indices = [0] * pts.size(0) + [1] * 1
            #marker.visualize(translations=pts) #marker_indices=marker_indices
            # env stepping
            #state, actions, samples = policy(obs)
            #obs, _, _, _ = env.step(actions.view(1, 2))
            #pose = torch.zeros(1, 2, device=env.unwrapped.device)
            #pose[:, :] = -1
            obs, _, _, _ = env.step(action)
            print(obs[:, :3])
            print()
            env.unwrapped.render()
        if not step % 50: print("Step {}/{}...".format(step, args_cli.video_length))
        if step >= args_cli.video_length-1: break
        if env.unwrapped.sim.is_stopped(): break
        step += 1

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
