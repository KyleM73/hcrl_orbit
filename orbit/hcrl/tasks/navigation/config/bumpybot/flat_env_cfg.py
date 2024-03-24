from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
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
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.orbit.envs.mdp as mdp
import orbit.hcrl.tasks.navigation.mdp as hcrl_mdp

##
# Pre-defined configs
##
from orbit.hcrl.assets import BUMPYBOT_CFG, GO1_CFG, CUBE_CFG, HOSPITAL_CFG  # isort: skip
from .navigation_env_cfg import LocomotionNavigationFlatEnvCfg

##
# Scene definition
##

@configclass
class BumpybotFlatEnvCfg(LocomotionNavigationFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = BUMPYBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.ground = HOSPITAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

        # turn off contact sensors
        self.scene.contactf_forces = None

        # reduce action scale
        self.actions.velocity.scale = (0.5,0.5,0.5)
        self.actions.velocity.offset = (0.0,0.0,0.0)

@configclass
class BumpybotFlatEnvCfg_PLAY(BumpybotFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        #self.randomization.base_external_force_torque = None
        #self.randomization.push_robot = None
        # viewer settings
        #self.viewer.eye = (-4.0, 0.0, 2.5)
        #self.viewer.lookat = (0.0, 0.0, 0.0)
