from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import BaseEnv

def root_pos_w(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body position in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.squeeze(1)

def body_pos_w(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body position in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1)

def body_heading_w(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body heading in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1))
    return math_utils.wrap_to_pi(yaw.unsqueeze(0))

def body_lin_vel_w(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body linear velocity in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)


def body_ang_vel_w(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body angular velocity in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)