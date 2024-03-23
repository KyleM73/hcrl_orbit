# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.envs import ViewerCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.orbit.envs.mdp as mdp
import orbit.hcrl.tasks.navigation.mdp as hcrl_mdp

##
# Pre-defined configs
##
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    #ground: AssetBaseCfg = MISSING
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    #obj: RigidObjectCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    se2_pose = hcrl_mdp.TrajectoryCommandCfg(
        asset_name="robot",
        body_name="dummy_revolute_yaw_link",
        resampling_time_range=(10.0, 15.0),
        simple_heading=True,
        normalized=False,
        threshold=0.5,
        ranges=hcrl_mdp.TrajectoryCommandCfg.Ranges(
            pos_x=(1.0, 2.0), pos_y=(1.0, 2.0), heading=(-0.0, 0.0) #heading unused if simple_heading==True
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    velocity = hcrl_mdp.HolonomicActionCfg(
        asset_name="robot",
        x_joint_name=["dummy_prismatic_x_joint"],
        y_joint_name=["dummy_prismatic_y_joint"],
        yaw_joint_name=["dummy_revolute_yaw_joint"],
        body_name=["dummy_revolute_yaw_link"],
        scale=(0.1,0.1,0.1), offset=(0.0,0.0,0.0),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        se2_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "se2_pose"}) # body frame
        #root_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        #yaw = ObsTerm(
        #    func=mdp.joint_pos_rel,
        #    params={"asset_cfg": SceneEntityCfg("robot", joint_names="dummy_revolute_yaw_joint")},
        #    noise=Unoise(n_min=-0.1, n_max=0.1)
        #)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    add_base_mass = EventTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dummy_revolute_yaw_link"), "mass_range": (-1.0, 1.0)},
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    #pose_potential_tracking_cls = mdp.pose_potential_tracking()
    #pose_potential_tracking = RewTerm(
    #    func=mdp.pose_potential_tracking, weight=10.0,
    #    params={
    #        "command_name": "se2_pose",
    #        "body_name": "dummy_revolute_yaw_link",
    #        "threshold" : 0.01,
    #        "asset_cfg": SceneEntityCfg("robot"),
    #        })
    heading_tracking_exp = RewTerm(
        func=hcrl_mdp.heading_tracking_exp, weight=1.0,
        params={"command_name": "se2_pose", "body_name": "dummy_revolute_yaw_link", "std": math.pi**0.5})
    #goal_reached = RewTerm(
    #    func=mdp.position_goal_reached_bonus, weight=0.1,
    #    params={"command_name": "se2_pose", "threshold": 0.5, "bonus": 100.0})
    pose_tracking_exp = RewTerm(
        func=hcrl_mdp.pose_tracking_exp, weight=1.0,
        params={"command_name": "se2_pose", "std": 2.0**0.5})
    #pose_tracking_inv = RewTerm(
    #    func=mdp.pose_tracking_inv, weight=1.0,
    #    params={"command_name": "se2_pose", "scale": 0.5})
    # -- penalties
    #ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    #dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    #action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # -- optional penalties
    #dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=hcrl_mdp.position_goal_reached, params={"command_name": "se2_pose", "threshold": 0.2}, time_out=True)
    #speed_limit = DoneTerm(func=mdp.velocity_limit, params={"body_name": "dummy_revolute_yaw_link", "threshold": 2.0})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

##
# Environment configuration
##

@configclass
class LocomotionNavigationFlatEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(eye=(7.5,7.5,7.5),origin_type="env")
    #viewer: ViewerCfg = ViewerCfg(eye=(3.0, 3.0, 3.0), origin_type="asset_root", asset_name="robot")
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt