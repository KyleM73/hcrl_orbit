from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs.rl_env import RLEnv

def reset_in_range(
        env: RLEnv, 
        env_ids: torch.Tensor, 
        asset_cfg: SceneEntityCfg, 
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        ):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint states
    joint_pos_min = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_pos_max = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_vel_min = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_vel_max = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip position to range
    joint_pos_lim = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_pos_min = torch.clamp(joint_pos_min, min=joint_pos_lim[..., 0], max=joint_pos_lim[..., 1])
    joint_pos_max = torch.clamp(joint_pos_max, min=joint_pos_lim[..., 0], max=joint_pos_lim[..., 1])
    abs_joint_vel_lim = asset.data.soft_joint_vel_limits[env_ids]
    # clip velocity to range
    joint_vel_min = torch.clamp(joint_vel_min, min=-abs_joint_vel_lim, max=abs_joint_vel_lim)
    joint_vel_max = torch.clamp(joint_vel_max, min=-abs_joint_vel_lim, max=abs_joint_vel_lim)
    # apply uniform random sample
    joint_pos = sample_uniform(joint_pos_min, joint_pos_max, joint_pos_min.shape, joint_pos_min.device)
    joint_vel = sample_uniform(joint_vel_min, joint_vel_max, joint_vel_min.shape, joint_vel_min.device)
    # write to the simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)