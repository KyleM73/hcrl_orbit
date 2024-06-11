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
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
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
from omni.isaac.orbit.utils.math import wrap_to_pi

import orbit.hcrl  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

if args_cli.plot:
    import matplotlib.pyplot as plt

def PDcontroller(state, obs, device, P_vel=1, P_theta=1):
    pose_d, _, theta_d = state[:, :2, 0], state[:, 2, 0], state[:, 3, 0]
    pose_diff = pose_d - obs[:, :2]
    theta_d = torch.arctan2(pose_diff[:, 1], pose_diff[:, 0])
    #theta_err = wrap_to_pi(theta_d - obs[:, 3])
    theta_err = theta_d - obs[:, 3]
    return torch.tensor([
        P_vel * pose_diff[:, 0] * torch.cos(theta_d),
        P_vel * pose_diff[:, 1] * torch.sin(theta_d),
        P_theta * theta_err
        ], device=device).view(-1, 3)

def PDcontrollerIntegrator(state, obs, device, P_vel=1, P_theta=1):
    pose_d, _, theta_d = state[:, :2, 0], state[:, 2, 0], state[:, 3, 0]
    pose_diff = pose_d - obs[:, :2]
    theta_d = torch.arctan2(pose_diff[:, 1], pose_diff[:, 0])
    theta_err = wrap_to_pi(theta_d - obs[:, 3])
    return torch.tensor([
        P_vel * pose_diff[:, 0], #* torch.cos(theta_d),
        P_vel * pose_diff[:, 1], #* torch.sin(theta_d),
        P_theta * theta_err
        ], device=device).view(-1, 3)


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
    obs[0, :2] = -4
    #obs[0, 0] = -3
    #obs[0, 1] = 3

    traj_pts = obs[:, :3].clone()
    dt = 0.1
    decimation = int(dt / env.unwrapped.step_dt)
    T = args_cli.video_length * dt / decimation
    viz_frac = 1
    #box_radius = 0.5/2 + 0.4 #c-space
    box_radii = [[0.5, 0.75],[1.0, 0.75]]
    #box_radii = [[1.0, 1.0]]
    #box_radii = [[0.75, 0.75], [0.75, 0.75]]
    policy = orbit.hcrl.tasks.navigation.pic.PathIntegralController(
        obs, dt, T, num_samples=10_000, border_radius=5.0,
        box_radius=box_radii, c_space_offset=0.2, 
        model="unicycle", device=env.unwrapped.device
    )

    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/testMarkers",
        markers={
            "sample": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.0,
                    metallic=0.0,
                    opacity=1.0
                    ),
            ),
            "trajectory": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.0,
                    metallic=0.0,
                    opacity=1.0
                ),
            ),
            "state": sim_utils.SphereCfg(
                radius=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.0,
                    metallic=0.0,
                    opacity=1.0
                ),
            )
        }
    )
    marker = VisualizationMarkers(cfg)
    
    # simulate environment
    while simulation_app.is_running():
        if not step % 50: print("Step {}/{}...".format(step, args_cli.video_length))
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            state, _, samples = policy(obs)

            if state.size(1) < 4:
                state = torch.cat((state, torch.zeros(1, 4 - state.size(1), 1, device=env.unwrapped.device)), dim=1)
            # visualize samples
            #sample_pts = samples[-1:, ::viz_frac, :3, 0] #sample every 100th trajectory
            sample_pts = samples[-1:, ::viz_frac, :3, 0] #sample every 100th trajectory
            sample_pts = sample_pts.flatten(0, 1)
            if sample_pts.size(1) < 3:
                sample_pts = torch.cat((sample_pts, torch.zeros(sample_pts.size(0), 1, device=env.unwrapped.device)), dim=1)
            #print(sample_pts.size())
            traj_pts = torch.cat((traj_pts, obs[:, :3]), dim=0)
            #pts = torch.cat((sample_pts, traj_pts, state[:, :3, 0]), dim=0)
            pts = torch.cat((sample_pts, traj_pts), dim=0)
            pts[:, 2] = 0.3
            marker_indices = [0] * sample_pts.size(0) + [1] * traj_pts.size(0)# + [2] * 1
            # env stepping
            print("state ",state)
            for _ in range(decimation):
                action = PDcontroller(state, obs, env.unwrapped.device, P_vel=1, P_theta=1) #10
                #action = PDcontrollerIntegrator(state, obs, env.unwrapped.device, P_vel=1, P_theta=1)
                obs, _, _, _ = env.step(action)
                #print("obs ",obs[0, :2])
                marker.visualize(translations=pts, marker_indices=marker_indices)
                env.unwrapped.render()
                step += 1
            collision_flag = policy.check_collision(obs, state[:, 0, :], state[:, 1, :], collision_flag=1.0)
            if collision_flag < 0.9:
                print("collision")
                break
            if step+1 >= args_cli.video_length: break
            if env.unwrapped.sim.is_stopped(): break
            #if collision_flag < 0.9: break
    print("Step {}/{}...".format(step, args_cli.video_length))
        
    # close the simulator
    env.close()
    print("Video saved to ",log_dir)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
