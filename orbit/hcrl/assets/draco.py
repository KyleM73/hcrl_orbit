"""Configuration for Draco robots.

The following configurations are available:

* :obj:`DRACO_CFG`: HCRL Apptronik Draco robot with simple PD controller for the legs

Reference: https://github.com/shbang91/rpc/blob/main/robot_model/draco/draco_modified.urdf
Reference: https://github.com/shbang91/rpc/blob/main/config/draco/nodelet.yaml
"""

from __future__ import annotations
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from isaac.lab.hcrl import EXT_DIR

##
# Configuration
##

ActuatorCfg = IdealPDActuatorCfg #ImplicitActuatorCfg

DRACO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/draco/draco.urdf")),
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "[lr]_hip_ie.*" : 0.000,
            "l_hip_aa.*" : 0.160,
            "r_hip_aa.*" : -0.160,
            "[lr]_hip_fe.*" : -1.1, #-0.785,
            "[lr]_knee_fe_jp.*" : 0.785,
            "[lr]_knee_fe_jd.*" : 1.1, #0.785,
            "[lr]_ankle_fe.*" : -1.1, #-0.785,
            "l_ankle_ie.*" : -0.160,
            "r_ankle_ie.*" : 0.160,
            "[lr]_shoulder_fe.*" : 0.000,
            "l_shoulder_aa.*" : 0.523,
            "r_shoulder_aa.*" : -0.523,
            "[lr]_shoulder_ie.*" : 0.000,
            "[lr]_elbow_fe.*" : -1.570,
            "[lr]_wrist_ps.*" : 0.000,
            "[lr]_wrist_pitch.*" : 0.000,
            "neck_pitch.*" : 0.000,
        },
        joint_vel={".*": 0.000},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "pitch": ActuatorCfg(
            joint_names_expr=["neck_pitch.*", "[lr]_wrist_pitch.*"],
            effort_limit=8.80,
            velocity_limit=76.0,
            stiffness={
                "neck_pitch.*": 15.0,
                "[lr]_wrist_pitch.*": 7.0,
            },
            damping={
                "neck_pitch.*": 0.0,
                "[lr]_wrist_pitch.*": 0.07,
            },
            friction={
                "neck_pitch.*": 0.0,
                "[lr]_wrist_pitch.*": 0.0
            },
        ),
         "shoulder": ActuatorCfg(
            joint_names_expr=["[lr]_shoulder.*"],
            effort_limit=18.0,
            velocity_limit=47.0,
            stiffness={
                "[lr]_shoulder_fe.*": 250.0,
                "[lr]_shoulder_aa.*": 300.0,
                "[lr]_shoulder_ie.*": 60.0,
            },
            damping={
                "[lr]_shoulder_fe.*": 3.0,
                "[lr]_shoulder_aa.*": 3.0,
                "[lr]_shoulder_ie.*": 2.0,
            },
            friction={
                "[lr]_shoulder.*": 0.0,
            },
         ),
        "elbow": ActuatorCfg(
            joint_names_expr=["[lr]_elbow_fe.*"],
            effort_limit=10.0,
            velocity_limit=59.2,
            stiffness={
                "[lr]_elbow_fe.*": 100.0,
            },
            damping={
                "[lr]_elbow_fe.*": 1.2,
            },
            friction={
                "[lr]_elbow_fe.*": 0.0,
            },
        ),
        "wrist": ActuatorCfg(
            joint_names_expr=["[lr]_wrist_ps.*"],
            effort_limit=12.0,
            velocity_limit=71.0,
            stiffness={
                "[lr]_wrist_ps.*": 15.0,
            },
            damping={
                "[lr]_wrist_ps.*": 1.0,
            },
            friction={
                "[lr]_wrist_ps.*": 0.0,
            },
        ),
        "hip_ie_ankle_fe": ActuatorCfg(
            joint_names_expr=["[lr]_hip_ie.*", "[lr]_ankle_fe.*"],
            effort_limit=44.0,
            velocity_limit=8.1,
            stiffness={
                "[lr]_hip_ie.*": 150.0,
                "[lr]_ankle_fe.*": 200.0, #150.0,
            },
            damping={
                "[lr]_hip_ie.*": 1.0,
                "[lr]_ankle_fe.*": 2.0, #0.2,
            },
            friction={
                "[lr]_hip_ie.*": 0.0,
                "[lr]_ankle_fe.*": 0.0,
            },
        ),
        "hip_aa": ActuatorCfg(
            joint_names_expr=["[lr]_hip_aa.*"],
            effort_limit=56.0,
            velocity_limit=5.1,
            stiffness={
                "[lr]_hip_aa.*": 450.0,
            },
            damping={
                "[lr]_hip_aa.*": 5.0,
            },
            friction={
                "[lr]_hip_aa.*": 0.0,
            },
        ),
        "hip_fe": ActuatorCfg(
            joint_names_expr=["[lr]_hip_fe.*"],
            effort_limit=59.6,
            velocity_limit=36.7,
            stiffness={
                "[lr]_hip_fe.*": 500.0,
            },
            damping={
                "[lr]_hip_fe.*": 5.0,
            },
            friction={
                "[lr]_hip_fe.*": 0.0,
            },
        ),
        "knee_fe": ActuatorCfg(
            joint_names_expr=["[lr]_knee_fe.*"],
            effort_limit=40.85,
            velocity_limit=13.35,
            stiffness={
                "[lr]_knee_fe.*": 450.0, #450.0, # right: 450 difference from hardware
            },
            damping={
                "[lr]_knee_fe.*": 5.0, #1.5,
            },
            friction={
                "[lr]_knee_fe.*": 0.0,
            },
        ),
        "ankle_ie": ActuatorCfg(
            joint_names_expr=["[lr]_ankle_ie.*"],
            effort_limit=30.0,
            velocity_limit=11.1,
            stiffness={
                "[lr]_ankle_ie.*": 30.0,
            },
            damping={
                "[lr]_ankle_ie.*": 0.1,
            },
            friction={
                "[lr]_ankle_ie.*": 0.0,
            },
        ),
    }
)
"""
Ordered Joints:
0:  'l_hip_ie',
1:  'l_shoulder_fe',
2:  'neck_pitch',
3:  'r_hip_ie',
4:  'r_shoulder_fe',
5:  'l_hip_aa',
6:  'l_shoulder_aa',
7:  'r_hip_aa',
8:  'r_shoulder_aa',
9:  'l_hip_fe',
10: 'l_shoulder_ie',
11: 'r_hip_fe',
12: 'r_shoulder_ie',
13: 'l_knee_fe_jp',
14: 'l_elbow_fe',
15: 'r_knee_fe_jp',
16: 'r_elbow_fe',
17: 'l_knee_fe_jd',
18: 'l_wrist_ps',
19: 'r_knee_fe_jd',
20: 'r_wrist_ps',
21: 'l_ankle_fe',
22: 'l_wrist_pitch',
23: 'r_ankle_fe',
24: 'r_wrist_pitch',
25: 'l_ankle_ie',
26: 'r_ankle_ie']
"""