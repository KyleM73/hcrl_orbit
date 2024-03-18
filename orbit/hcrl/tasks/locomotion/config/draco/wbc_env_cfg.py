from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg, ViewerCfg
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

import omni.isaac.orbit.envs.mdp as mdp
import orbit.hcrl.tasks.locomotion.mdp as hcrl_mdp

##
# Pre-defined configs
##
#from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from orbit.hcrl.assets.draco import DRACO_CFG  # isort: skip


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
    robot: ArticulationCfg = DRACO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = None
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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

    pass

    """base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.0, 0.0), heading=(-0*math.pi, 0*math.pi)
        ),
    )"""

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["^(?!.*knee_fe_jd$).*"], scale=0.0, use_default_offset=True) #jp -> jd
    joint_torque = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["^(?!.*knee_fe_jd$).*"], scale=1.0)
    # TODO: make custom WBC action cfg (WBC in self.apply_action())
    #"^(?!.*knee_fe_jd$).*"


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class BaseComPosCfg(ObsGroup):
        base_com_pos = ObsTerm(func=mdp.root_pos_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseComQuatCfg(ObsGroup):
        base_com_quat = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseComLinVelCfg(ObsGroup):
        base_com_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseComAngVelCfg(ObsGroup):
        base_com_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseJointPosCfg(ObsGroup):
        base_joint_pos = ObsTerm(func=mdp.root_pos_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseJointQuatCfg(ObsGroup):
        base_joint_quat = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseJointLinVelCfg(ObsGroup):
        base_joint_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class BaseJointAngVelCfg(ObsGroup):
        base_joint_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class JointPosCfg(ObsGroup):
        joint_pos = ObsTerm(func=hcrl_mdp.joint_pos)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class JointVelCfg(ObsGroup):
        joint_vel = ObsTerm(func=mdp.joint_vel_rel) # only works because default vel is zero
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class RfContactCfg(ObsGroup):
        b_rf_contact = ObsTerm(func=hcrl_mdp.in_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="r_ankle_ie_link"), "threshold": 1.0
        })
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class LfContactCfg(ObsGroup):
        b_lf_contact = ObsTerm(func=hcrl_mdp.in_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="l_ankle_ie_link"), "threshold": 1.0
        })
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        #projected_gravity = ObsTerm(
        #    func=mdp.projected_gravity,
        #    noise=Unoise(n_min=-0.05, n_max=0.05),
        #)
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        #actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    base_com_pos: BaseComPosCfg = BaseComPosCfg()
    base_com_quat: BaseComQuatCfg = BaseComQuatCfg()
    base_com_lin_vel: BaseComLinVelCfg = BaseComLinVelCfg()
    base_com_ang_vel: BaseComAngVelCfg = BaseComAngVelCfg()
    base_joint_pos: BaseJointPosCfg = BaseJointPosCfg()
    base_joint_quat: BaseJointQuatCfg = BaseJointQuatCfg()
    base_joint_lin_vel: BaseJointLinVelCfg = BaseJointLinVelCfg()
    base_joint_ang_vel: BaseJointAngVelCfg = BaseJointAngVelCfg()
    joint_pos: JointPosCfg = JointPosCfg()
    joint_vel: JointVelCfg = JointVelCfg()
    b_rf_contact: RfContactCfg = RfContactCfg()
    b_lf_contact: LfContactCfg = LfContactCfg()

@configclass
class RandomizationCfg:
    """Configuration for randomization."""
    pass
    # startup
    """physics_material = RandTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    base_external_force_torque = RandTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = RandTerm(
        func=hcrl_mdp.reset_in_range,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )"""

    # interval


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass
    """time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[
                "torso_link",
                "neck_pitch.*", 
                "[lr]_shoulder_ie.*", 
                "[lr]_wrist_pitch.*",
                "[lr]_knee_fe.*",
                "[lr]_hip_fe.*"]
                ),
            "threshold": 1.0,
            },
    )"""


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class WBCEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Viewer
    viewer: ViewerCfg = ViewerCfg(eye=(3,3,3), origin_type="asset_root", asset_name="robot")
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.decimation = 1 #16
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.001 #0.00125
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

class WBCEnvCfg_PLAY(WBCEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None