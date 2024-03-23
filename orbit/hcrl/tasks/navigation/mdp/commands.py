from __future__ import annotations

from dataclasses import MISSING
import torch
from typing import TYPE_CHECKING, Sequence

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import CommandTermCfg
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import CommandTerm
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.orbit.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat, quat_from_euler_xyz, euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

class BoxPoseCommand(CommandTerm):
    """Command generator that generates position commands based on box constraints.

    The position commands are sampled from a uniform box with constraints and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    cfg: BoxPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BoxPoseCommandCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_id = self.robot.find_bodies(self.cfg.body_name)[0][0]
        if self.cfg.normalized: 
            # only normalize x-y pos
            self.norm = torch.tensor([self.cfg.ranges.pos_x[1], self.cfg.ranges.pos_y[1], 1.0], device=self.device)

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device) #x-y-z
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device) #yaw
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BoxPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        cmd = torch.cat((self.pos_command_b[:, :2],self.heading_command_b.view(-1, 1)),dim=-1)
        #TODO fix numerical issues with divide by zero
        if self.cfg.normalized:
            cmd /= self.norm
        return cmd

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        # TODO: need to add constraints
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        # offset the position command by the current root position
        self.pos_command_w[env_ids, :2] += self.robot.data.root_pos_w[env_ids, :2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.body_pos_w[env_ids, self.body_id, :]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_direction + torch.pi)

            _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[env_ids, self.body_id, :])
            self.heading_command_w[env_ids] = torch.where(
                wrap_to_pi(target_direction - yaw).abs()
                < wrap_to_pi(flipped_heading - yaw).abs(),
                target_direction,
                flipped_heading,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.body_pos_w[:, self.body_id, :]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.body_quat_w[:, self.body_id, :]), target_vec)
        _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.body_id, :])
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - yaw)

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.body_pos_w[:, self.body_id, :2], dim=1)
        _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.body_id, :])
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - yaw))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "box_heading_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/heading_goal"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.box_heading_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/heading_current"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.box_heading_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
            self.box_heading_goal_visualizer.set_visibility(True)
            self.box_heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if hasattr(self, "box_heading_goal_visualizer"):
                self.box_heading_goal_visualizer.set_visibility(False)
                self.box_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.box_goal_visualizer.visualize(self.pos_command_w)
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.body_pos_w[:, self.body_id, :].clone()
        base_pos_w[:, 2] += 0.75
        # display markers
        self.box_heading_goal_visualizer.visualize(base_pos_w, self._resolve_heading_to_arrow())
        self.box_heading_visualizer.visualize(base_pos_w, self.robot.data.body_quat_w[:, self.body_id, :])

    """
    Internal helpers.
    """

    def _resolve_heading_to_arrow(self) -> torch.Tensor:
        """Converts the heading command to arrow direction rotation."""
        # arrow-direction
        zeros = torch.zeros_like(self.heading_command_w)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, self.heading_command_w)
        return arrow_quat

@configclass
class BoxPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = BoxPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """
    normalized: bool = MISSING
    """Whether to normalize the command by the max range."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        # only use if simple_heading is False
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""






# ========================================================================================================
# ========================================================================================================
# ========================================================================================================






class TrajectoryCommand(CommandTerm):
    """Command generator that generates position commands based on box constraints.

    The position commands are sampled from a uniform box with constraints and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    cfg: TrajectoryCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TrajectoryCommandCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_id = self.robot.find_bodies(self.cfg.body_name)[0][0]
        if self.cfg.normalized: 
            # only normalize x-y pos
            self.norm = torch.tensor([self.cfg.ranges.pos_x[1], self.cfg.ranges.pos_y[1], 1.0], device=self.device)

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device) #x-y-z
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device) #yaw
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.switch = torch.tensor([-1, 1], device=self.device)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TrajectoryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        command = torch.cat((self.pos_command_b[:, :2], self.heading_command_b.view(-1, 1)), dim=-1)
        #TODO fix numerical issues with divide by zero
        if self.cfg.normalized:
            command /= self.norm
        return command

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x) * self.switch[torch.randint(2,(len(env_ids),))]
        self.pos_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y) * self.switch[torch.randint(2,(len(env_ids),))]
        # offset the position command by the current root position
        self.pos_command_w[env_ids, :2] += self.robot.data.body_pos_w[env_ids, self.body_id, :2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.body_pos_w[env_ids, self.body_id, :]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_direction + torch.pi)

            _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[env_ids, self.body_id, :])
            self.heading_command_w[env_ids] = torch.where(
                wrap_to_pi(target_direction - yaw).abs()
                < wrap_to_pi(flipped_heading - yaw).abs(),
                target_direction,
                flipped_heading,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        # resample when near target
        #env_ids = (torch.linalg.norm(self.pos_command_b, dim=-1) < self.cfg.threshold).nonzero().flatten()
        #self._resample_command(env_ids)
        print("pose: ",self.robot.data.body_pos_w[:, self.body_id, :])
        print("velocity: ",self.robot.data.body_lin_vel_w[:, self.body_id, :])
        target_vec = self.pos_command_w - self.robot.data.body_pos_w[:, self.body_id, :]
        target_vec[:, 2] = 0
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.body_quat_w[:, self.body_id, :]), target_vec)
        _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.body_id, :])
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - yaw)
        print("pos comand: ",self.pos_command_b)
        print("heading_command: ",self.heading_command_b)

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.body_pos_w[:, self.body_id, :2], dim=1)
        _,_,yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self.body_id, :])
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - yaw))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "box_heading_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/heading_goal"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.box_heading_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/heading_current"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.box_heading_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.box_goal_visualizer.set_visibility(True)
            self.box_heading_goal_visualizer.set_visibility(True)
            self.box_heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if hasattr(self, "box_heading_goal_visualizer"):
                self.box_heading_goal_visualizer.set_visibility(False)
                self.box_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.box_goal_visualizer.visualize(self.pos_command_w)
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.body_pos_w[:, self.body_id, :].clone()
        base_pos_w[:, 2] += 0.75
        # display markers
        self.box_heading_goal_visualizer.visualize(base_pos_w, self._resolve_heading_to_arrow())
        self.box_heading_visualizer.visualize(base_pos_w, self.robot.data.body_quat_w[:, self.body_id, :])

    """
    Internal helpers.
    """

    def _resolve_heading_to_arrow(self) -> torch.Tensor:
        """Converts the heading command to arrow direction rotation."""
        # arrow-direction
        zeros = torch.zeros_like(self.heading_command_w)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, self.heading_command_w)
        return arrow_quat
    
@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = TrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """
    normalized: bool = MISSING
    """Whether to normalize the command by the max range."""

    threshold: float = MISSING
    """Threshold at which point to resample trajectory waypoints."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        # only use if simple_heading is False
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""