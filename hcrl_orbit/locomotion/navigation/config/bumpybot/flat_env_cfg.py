from omni.isaac.orbit.utils import configclass

from hcrl_orbit.locomotion.navigation.navigation_env_cfg import LocomotionNavigationFlatEnvCfg

##
# Pre-defined configs
##
from hcrl_orbit.assets.hcrl_robots.bumpybot import BUMPYBOT_CFG  # isort: skip


@configclass
class BumpybotFlatEnvCfg(LocomotionNavigationFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = BUMPYBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # reduce action scale
        #self.actions.joint_pos.scale = 0.0

        # randomization
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-0.0, 0.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.randomization.reset_robot_joints.params["position_range"] = (-0.0, 0.0)
        self.randomization.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0, 0), "pitch": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        #self.randomization.push_robot.params["velocity_range"] = {"x": (-0.0, 0.0), "y": (-0.0, 0.0)}

        # rewards
        #self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_ie_link"
        #self.rewards.feet_air_time.weight = 0.01
        #self.rewards.undesired_contacts = None
        #self.rewards.dof_torques_l2.weight = -0.0002
        #self.rewards.track_lin_vel_xy_exp.weight = 1.5
        #self.rewards.track_ang_vel_z_exp.weight = 0.75
        #self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        #self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class BumpybotFlatEnvCfg_PLAY(BumpybotFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
