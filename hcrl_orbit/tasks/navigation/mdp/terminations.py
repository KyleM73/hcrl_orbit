from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

"""
MDP terminations.
"""

def position_goal_reached(
        env: RLTaskEnv, 
        command_name: str,
        threshold: float = 0.5,
    ) -> torch.Tensor:
    """Terminate the episode when the robot reaches its goal state."""
    goal_pose_body = env.command_manager.get_command(command_name)[:, :2] # body pose command
    return torch.linalg.norm(goal_pose_body,dim=-1) < threshold

def velocity_limit(
        env: RLTaskEnv, 
        body_name: str,
        threshold: float = 2.0,
    ) -> torch.Tensor:
    """Terminate the episode when the robot violates the speed limit."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    body_id = asset.find_bodies(body_name)[0][0]
    return torch.linalg.norm(asset.data.body_lin_vel_w[:, body_id, :2], dim=-1) > threshold