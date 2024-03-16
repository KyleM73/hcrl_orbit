"""Configuration for Go1 robot.

The following configurations are available:

* :obj:`Go1_CFG`: Unitree Go1 robot with simple PD controller for the legs

Reference: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go1_description/urdf/go1.urdf
Reference: https://github.com/Improbable-AI/walk-these-ways/tree/master/resources/robots/go1/meshes
"""

from __future__ import annotations
import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import IdealPDActuatorCfg, DCMotorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg

from orbit.hcrl import EXT_DIR

##
# Configuration
##

GO1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/go1/go1.urdf")),
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_uleg_joint": 0.8,
            "R[L,R]_uleg_joint": 1.0,
            ".*_lleg_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=100, #25
            damping=5, #0.5
            friction=0.2,
        ),
        "uleg": DCMotorCfg(
            joint_names_expr=[".*_uleg_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=300, #25
            damping=8, #0.5
            friction=0.2,
        ),
        "lleg": DCMotorCfg(
            joint_names_expr=[".*_lleg_joint"],
            effort_limit=35.55,
            saturation_effort=35.55,
            velocity_limit=20.06,
            stiffness=300, #25
            damping=8, #0.5
            friction=0.2,
        ),
    },
)
"""Configuration of Unitree Go1 using DC-Motor actuator model."""