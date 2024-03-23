"""Configuration for Draco robots.

The following configurations are available:

* :obj:`BUMPYBOT_CFG`: HCRL Bumpybot robot with Nonholonomic controller

Reference: TODO add link to urdf etc.
"""

from __future__ import annotations
import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg, RigidObjectCfg

from orbit.hcrl import EXT_DIR

##
# Configuration
##

CUBE_CFG = RigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)

ActuatorCfg = IdealPDActuatorCfg #ImplicitActuatorCfg

BUMPYBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/bumpybot.urdf")),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={".*" : 0.000},
        joint_vel={".*": 0.000},
    ),
    #soft_joint_pos_limit_factor=1.0,
    actuators={
        "prismatic": ActuatorCfg(
            joint_names_expr=["dummy_prismatic.*"],
            effort_limit=1000.0,
            velocity_limit=100.0,
            stiffness={"dummy_prismatic.*": 0}, #2e-3
            damping={"dummy_prismatic.*": 1}, #1e-5
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": ActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=100.0,
            stiffness={"dummy_revolute.*": 0},
            damping={"dummy_revolute.*": 1},
            friction={"dummy_revolute.*": 0.0},
         ),
    }
)

BUMPYBOT_OBJECT_CFG = RigidObjectCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/bumpybot_object.urdf")),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
    ),
    collision_group=0,
)