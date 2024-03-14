"""Script to record simulations"""

from __future__ import annotations

import argparse
import os

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video_length", type=int, default=200, help="Length to record the video (in control steps)")
parser.add_argument("--plot", action="store_true", default=False, help="Plot joint angles")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
import hcrl_orbit # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)
if args_cli.plot:
    import matplotlib.pyplot as plt

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: RLTaskEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # TODO #print(env.unwrapped.scene["robot"].joint_names)
    # TODO #print(env.unwrapped.scene["robot"].data.default_joint_pos)
    # wrap for video recording
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos"),
        "step_trigger": lambda step: step % 1000000000 == 0,
        "video_length": args_cli.video_length,
        "disable_logger": False,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env.metadata["video.frames_per_second"] = 1.0 / env.unwrapped.step_dt
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # export policy to onnx
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_onnx(runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    step = 0
    if args_cli.plot:
        q = []
        qdot = []
    # to mitigate the first-few-frames-black problem (see orbit known issues)
    for _ in range(100):
        env.unwrapped.sim.render()
    # simulate environment
    while simulation_app.is_running():
        #print(obs[0, :])
        #print("l_knee_fe_jp: ",obs[0, 12+13])
        #print("l_knee_fe_jd: ",obs[0, 12+17])
        #print("r_knee_fe_jp: ",obs[0, 12+15])
        #print("r_knee_fe_jd: ",obs[0, 12+19])
        #print()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            env.unwrapped.render()
        if args_cli.plot:
            q.append(obs[0, 0:3].view(1,3).cpu()) #command
            qdot.append(obs[0, 3:6].view(1,3).cpu())
            #q.append(obs[0, 12:12+27].view(1,27).cpu())
            #qdot.append(obs[0, 12+27:12+27+27].view(1,27).cpu())
        if not step % 50: print("Step {}/{}...".format(step, args_cli.video_length))
        if step > args_cli.video_length:
            break
        step += 1
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()

    if args_cli.plot:
        q = torch.cat(q, dim=0)
        qdot = torch.cat(qdot, dim=0)

        fig,ax = plt.subplots(2)
        joints = [
            ["x", 0],
            ["y", 1],
            ["yaw", 2],
        ]
        """joints = [
            ["l_hip_fe",  9],
            ["r_hip_fe", 11],
            ["l_knee_fe_jp", 13],
            ["r_knee_fe_jp", 15],
            ["l_knee_fe_jd", 17],
            ["r_knee_fe_jd", 19],
            ["l_ankle_fe", 21],
            ["r_ankle_fe", 23],
            ]"""
        for name, i in joints:
            ax[0].plot(q[:,i],label=name) 
            ax[1].plot(qdot[:,i],label=name)
        
        ax[0].legend()
        ax[1].legend()
        plt.savefig(log_dir+"/q.png", dpi=150)

if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()