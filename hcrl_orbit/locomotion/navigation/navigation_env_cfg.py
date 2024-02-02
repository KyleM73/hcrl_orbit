# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import hcrl_orbit.locomotion.navigation.mdp as mdp

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
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
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
    # TODO write new command generator with max distance constraints (eg in -10,10)
    base_pose = mdp.BoxPoseCommandCfg(
        asset_name="robot",
        body_name="dummy_revolute_yaw_link",
        resampling_time_range=(3.0, 5.0),
        simple_heading=True,
        normalized=True,
        ranges=mdp.BoxPoseCommandCfg.Ranges(
            pos_x=(-20.0, 20.0), pos_y=(-20.0, 20.0), heading=(-0.0, 0.0) #heading unused if simple_heading==True
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # TODO dont we want this to be holonomic? write new action class
    """se2_pose = mdp.HolonomicActionCfg(
        asset_name="robot",
        x_joint_name=["dummy_prismatic_x_joint"],
        y_joint_name=["dummy_prismatic_y_joint"],
        yaw_joint_name=["dummy_revolute_yaw_joint"],
        scale=(0,0,0), offset=(1.0,0.0,0.0) # TODO this may be too high
        )"""
    #se2_pose = mdp.JointEffortActionCfg(
    #    asset_name="robot",
    #    joint_names=[".*"],
    #    scale={"dummy_prismatic.*": 10.0,
    #           "dummy_revolute.*": 1.0,
    #           },
    #    offset={"dummy_prismatic.*": 0.0,
    #           "dummy_revolute.*": 0.0,
    #           },
    #    )
    se2_pose = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={"dummy_prismatic.*": 1.0,
               "dummy_revolute.*": 3.14,
               },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_pose"})
        joint_pos = ObsTerm(func=mdp.joint_pos_norm, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        #contact_wrench = ObsTerm(
        #    func=mdp.body_incoming_wrench, noise=Unoise(n_min=-0.01, n_max=0.01),
        #    params={"asset_cfg": SceneEntityCfg("robot", body_names="dummy_revolute.*")}
        #    )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    physics_material = RandTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 0.0), #0.8
            "dynamic_friction_range": (0.0, 0.0), #0.6
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = RandTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="dummy_revolute.*"), "mass_range": (-0.0, 0.0)},
    )

    # reset
    base_external_force_torque = RandTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="dummy_revolute.*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = RandTerm(
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
    push_robot = RandTerm(
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
    pose_potential_tracking = RewTerm(
        func=mdp.pose_potential_tracking, weight=10.0, params={
            "command_name": "base_pose",
            "body_name": "dummy_revolute_yaw_link",
            "threshold" : 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            })
    heading_tracking_exp = RewTerm(
        func=mdp.heading_tracking_exp, weight=1.0,
        params={"command_name": "base_pose", "body_name": "dummy_revolute_yaw_link", "std": 1.0})
    # -- penalties
    #ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    #dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # -- optional penalties
    #dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # TODO bad orientation


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass


##
# Environment configuration
##


@configclass
class LocomotionNavigationFlatEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
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
