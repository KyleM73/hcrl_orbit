from __future__ import annotations

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
#import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
#import hcrl_orbit.tasks.locomotion.mdp as hcrl_mdp

##
# Pre-defined configs
##
from orbit.hcrl.assets import GO1_CFG # isort: skip

@configclass
class Go1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # randomization
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "base"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.randomization.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.randomization.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base", ".*leg"]

        # viewer
        self.viewer.eye = (2.0,2.0,2.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class Go1RoughEnvCfg_PLAY(Go1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None