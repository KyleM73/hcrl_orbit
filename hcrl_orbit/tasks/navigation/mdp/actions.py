from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

import carb

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

class HolonomicAction(ActionTerm):
    r"""Holonomic action that maps a two dimensional action to the velocity of the robot in
    the x, y and yaw directions.

    This action term helps model a holonomic robot base. The action is a 3D vector which comprises of the
    forward velocity :math:`v_{B,x:y}` and the turning rate :\omega_{B,z}: in the base frame. Using the current
    base orientation, the commands are transformed into dummy joint velocity targets as:

    .. math::

        \dot{q}_{0, des} &= v_{B,x} \\
        \dot{q}_{1, des} &= v_{B,y} \\
        \dot{q}_{2, des} &= \omega_{B,z}

    Since the base is simulated as a dummy joint, the yaw is directly
    the value of the revolute joint along z, i.e., :math:`q_2 = \theta` where :math:`\theta` is the yaw of the 2-D base. 

    .. note::
        The current implementation assumes that the base is simulated with three dummy joints (prismatic joints along x
        and y, and revolute joint along z). This is because it is easier to consider the mobile base as a floating link
        controlled by three dummy joints, in comparison to simulating wheels which is at times is tricky because of
        friction settings.

        However, the action term can be extended to support other base configurations as well.

    .. tip::
        For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints.
        This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
    """

    cfg: HolonomicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 3)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 3)."""

    def __init__(self, cfg: HolonomicActionCfg, env: BaseEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse the joint information
        # -- x joint
        x_joint_id, x_joint_name = self._asset.find_joints(self.cfg.x_joint_name)
        if len(x_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the x joint name: {self.cfg.x_joint_name}, got {len(x_joint_id)}"
            )
        # -- y joint
        y_joint_id, y_joint_name = self._asset.find_joints(self.cfg.y_joint_name)
        if len(y_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the y joint name: {self.cfg.y_joint_name}")
        # -- yaw joint
        yaw_joint_id, yaw_joint_name = self._asset.find_joints(self.cfg.yaw_joint_name)
        if len(yaw_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the yaw joint name: {self.cfg.yaw_joint_name}")

        # process into a list of joint ids
        self._joint_ids = [x_joint_id[0], y_joint_id[0], yaw_joint_id[0]]
        self._joint_names = [x_joint_name[0], y_joint_name[0], yaw_joint_name[0]]
        # log info for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, 3, device=self.device)

        # save the scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        # compute joint velocities targets
        self._joint_vel_command[:, 0] = self.processed_actions[:, 0]  # x
        self._joint_vel_command[:, 1] = self.processed_actions[:, 1]  # y
        self._joint_vel_command[:, 2] = self.processed_actions[:, 2]  # yaw
        # set the joint velocity targets
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)



@configclass
class HolonomicActionCfg(ActionTermCfg):
    """Configuration for the holonomic action term with dummy joints at the base.

    See :class:`HolonomicAction` for more details.
    """

    class_type: type[ActionTerm] = HolonomicAction

    x_joint_name: str = MISSING
    """The dummy joint name in the x direction."""
    y_joint_name: str = MISSING
    """The dummy joint name in the y direction."""
    yaw_joint_name: str = MISSING
    """The dummy joint name in the yaw direction."""
    scale: tuple[float, float] = (1.0, 1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0, 1.0)."""
    offset: tuple[float, float] = (0.0, 0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0, 0.0)."""