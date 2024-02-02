from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from omni.isaac.orbit.sensors import ContactSensor
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def pose_tracking_exp(env: RLTaskEnv, command_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward pose command tracking
    """
    target_pose = env.command_manager.get_command(command_name)[:, :2] # x-y command
    pose = env.scene["robot"].data.root_pos_w[:, :2] # x-y pose
    pose_error = torch.linalg.norm(pose-target_pose, dim=-1)
    return torch.exp(-pose_error/std**2)

def heading_tracking_exp(env: RLTaskEnv, command_name: str, body_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward heading command tracking
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    body_id = asset.find_bodies(body_name)[0][0]
    target_heading = env.command_manager.get_command(command_name)[:, 2] # heading command
    _, _, heading = math_utils.euler_xyz_from_quat(env.scene["robot"].data.body_quat_w[:, body_id, :]) # heading
    # TODO wrap to pi
    #heading_error = torch.linalg.norm(heading-target_heading, dim=-1)
    heading_error = torch.abs(math_utils.wrap_to_pi(target_heading - heading))
    return torch.exp(-heading_error/std**2)

"""
class pose_potential_tracking:
    def __init__(self):
        self.last_distance_to_goal = torch.zeros(1, 1)
    def __call__(self, env: RLTaskEnv, command_name: str, threshold: float = 0.05) -> torch.Tensor:
        pose = env.scene["robot"].data.root_pos_w[:, :2] # x-y pose
        vel_norm = torch.linalg.norm(env.scene["robot"].data.root_lin_vel[:, :2], dim=-1)
        goal_pose = env.command_manager.get_command(command_name)[:, :2] # x-y command
        distance_to_goal = torch.linalg.norm(pose-goal_pose, dim=-1)
        potential = self.last_distance_to_goal - distance_to_goal
        potential = torch.where(vel_norm > threshold, potential/(vel_norm*env.step_dt), 0)
        self.last_distance_to_goal = distance_to_goal
        return torch.clamp(potential, -1, 1)
"""
    

class pose_potential_tracking(ManagerTermBase):
    """Reward for making progress towards the goal."""

    def __init__(self, env: RLTaskEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.last_distance_to_goal = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        self.body_id = asset.find_bodies(self.cfg.params["body_name"])[0][0]
        # compute projection of current heading to desired heading vector
        goal_pose = self._env.command_manager.get_command(self.cfg.params["command_name"])[env_ids, :2]
        distance_to_goal = torch.linalg.norm(goal_pose - asset.data.body_pos_w[env_ids, self.body_id, :2], dim=-1)
        vel_norm = torch.linalg.norm(asset.data.body_lin_vel_w[env_ids, self.body_id, :2], dim=-1)
        potential = self.last_distance_to_goal[env_ids] - distance_to_goal
        potential = torch.where(vel_norm > self.cfg.params["threshold"], potential/(vel_norm*self._env.step_dt), 0)
        self.last_distance_to_goal[env_ids] = distance_to_goal
        
    def __call__(
        self,
        env: RLTaskEnv,
        command_name: str, body_name: str, threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute projection of current heading to desired heading vector
        goal_pose = env.command_manager.get_command(command_name)[:, :2]
        distance_to_goal = torch.linalg.norm(goal_pose - asset.data.body_pos_w[:, self.body_id, :2], dim=-1)
        vel_norm = torch.linalg.norm(asset.data.body_lin_vel_w[:, self.body_id, :2], dim=-1)
        potential = self.last_distance_to_goal - distance_to_goal
        potential = torch.where(vel_norm > threshold, potential/(vel_norm*env.step_dt), 0)
        self.last_distance_to_goal = distance_to_goal
        # reward terms
        return torch.clamp(potential, -1, 1)