from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from omni.isaac.orbit.sensors import ContactSensor
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def pose_tracking_exp_l2(env: RLTaskEnv, command_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward pose command tracking
    """
    goal_pose_b = env.command_manager.get_command(command_name)[:, :2] # x-y command
    return torch.exp(-torch.sum(torch.square(goal_pose_b), dim=-1)/std**2)

def pose_tracking_exp_l1(env: RLTaskEnv, command_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward pose command tracking
    """
    goal_pose_b = env.command_manager.get_command(command_name)[:, :2] # x-y command
    return torch.exp(-torch.sum(goal_pose_b.abs(), dim=-1)/std**2)

def vel_dir_tracking_inv(env: RLTaskEnv, command_name: str, body_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward velocity direction tracking
    """
    goal_pose_b = env.command_manager.get_command(command_name)[:, :2] # x-y command
    pose_norm = torch.linalg.norm(goal_pose_b, dim=-1)
    goal_velocity_direction = torch.where(pose_norm > 1e-3, goal_pose_b / pose_norm, torch.zeros_like(goal_pose_b))

def heading_tracking_exp_l2(env: RLTaskEnv, command_name: str, body_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward heading command tracking
    """
    # extract the used quantities (to enable type-hinting)
    #asset: Articulation = env.scene["robot"]
    #body_id = asset.find_bodies(body_name)[0][0]
    goal_heading_b = env.command_manager.get_command(command_name)[:, 2] # body heading command
    return torch.exp(-torch.square(goal_heading_b)/std**2)

def heading_tracking_exp_l1(env: RLTaskEnv, command_name: str, body_name: str, std: float = 1.0) -> torch.Tensor:
    """
    Reward heading command tracking
    """
    # extract the used quantities (to enable type-hinting)
    #asset: Articulation = env.scene["robot"]
    #body_id = asset.find_bodies(body_name)[0][0]
    goal_heading_b = env.command_manager.get_command(command_name)[:, 2] # body heading command
    return torch.exp(-goal_heading_b.abs()/std**2)

def position_goal_reached_bonus(
        env: RLTaskEnv, 
        command_name: str,
        threshold: float = 0.5,
        bonus: float = 1.0,
    ) -> torch.Tensor:
    """Reward the robot when the robot reaches its goal state."""
    goal_pose_body = env.command_manager.get_command(command_name)[:, :2] # body pose command
    #dist_reward = torch.exp(-torch.linalg.norm(goal_pose_body,dim=-1))
    return torch.where(torch.linalg.norm(goal_pose_body,dim=-1) < threshold, bonus, 0.0)
    

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
        goal_pose = torch.linalg.norm(self._env.command_manager.get_command(self.cfg.params["command_name"])[env_ids, :2], dim=-1)
        #distance_to_goal = torch.linalg.norm(goal_pose - asset.data.body_pos_w[env_ids, self.body_id, :2], dim=-1)
        #vel_norm = torch.linalg.norm(asset.data.body_lin_vel_w[env_ids, self.body_id, :2], dim=-1)
        #potential = self.last_distance_to_goal[env_ids] - goal_pose
        #potential = torch.where(vel_norm > self.cfg.params["threshold"], potential/(vel_norm*self._env.step_dt), 0)
        self.last_distance_to_goal[env_ids] = goal_pose
        
    def __call__(
        self,
        env: RLTaskEnv,
        command_name: str, body_name: str, threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute projection of current heading to desired heading vector
        goal_pose = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
        #distance_to_goal = torch.linalg.norm(goal_pose - asset.data.body_pos_w[:, self.body_id, :2], dim=-1)
        vel_norm = torch.linalg.norm(asset.data.body_lin_vel_w[:, self.body_id, :2], dim=-1)
        potential = self.last_distance_to_goal - goal_pose
        potential = torch.where(vel_norm > threshold, potential/(vel_norm*env.step_dt), 0)
        self.last_distance_to_goal = goal_pose
        # reward terms
        return torch.clamp(potential, -1, 1)

def joint_velocity_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    speed = asset.data.joint_vel[:, asset_cfg.joint_ids].abs()
    velocity_penalty = torch.where(speed > threshold,
        speed - threshold,
        0,
    )
    return torch.sum(torch.square(velocity_penalty), dim=1)