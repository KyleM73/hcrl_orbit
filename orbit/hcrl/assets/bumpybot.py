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

BUMPYBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        activate_contact_sensors=False,
        self_collision=False,
        convex_decompose_mesh=True,
        link_density=1e-5,
        visible=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/bumpybot.urdf")),
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
        pos=(0.0, 0.0, 0.343),
        joint_pos={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 3.141},
        joint_vel={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 0.000},
    ),
    soft_joint_pos_limit_factor=1.0,
    collision_group=0,
    actuators={
        "prismatic": IdealPDActuatorCfg(
            joint_names_expr=["dummy_prismatic.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_prismatic.*": 0},
            damping={"dummy_prismatic.*": 1000},
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": IdealPDActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_revolute.*": 0},
            damping={"dummy_revolute.*": 1000},
            friction={"dummy_revolute.*": 0.0},
         ),
         #"passive": IdealPDActuatorCfg(
         #   joint_names_expr=["passive_prismatic_z_joint"],
         #   effort_limit=0.0,
         #   velocity_limit=10.0,
         #   stiffness={"passive_prismatic_z_joint": 0},
         #   damping={"passive_prismatic_z_joint": 0},
         #   friction={"passive_prismatic_z_joint": 0},
         #)
    },
    debug_vis=True,
)

UNICYCLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        activate_contact_sensors=False,
        self_collision=False,
        convex_decompose_mesh=True,
        link_density=1e-5,
        visible=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/unicycle.urdf")),
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
        pos=(0.0, 0.0, 0.343),
        joint_pos={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 3.141},
        joint_vel={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 0.000},
    ),
    soft_joint_pos_limit_factor=1.0,
    collision_group=0,
    actuators={
        "prismatic": IdealPDActuatorCfg(
            joint_names_expr=["dummy_prismatic.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_prismatic.*": 0},
            damping={"dummy_prismatic.*": 1000},
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": IdealPDActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_revolute.*": 0},
            damping={"dummy_revolute.*": 1000},
            friction={"dummy_revolute.*": 0.0},
         ),
         #"passive": IdealPDActuatorCfg(
         #   joint_names_expr=["passive_prismatic_z_joint"],
         #   effort_limit=0.0,
         #   velocity_limit=10.0,
         #   stiffness={"passive_prismatic_z_joint": 0},
         #   damping={"passive_prismatic_z_joint": 0},
         #   friction={"passive_prismatic_z_joint": 0},
         #)
    },
    debug_vis=True,
)

CAR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        activate_contact_sensors=False,
        self_collision=False,
        convex_decompose_mesh=True,
        link_density=1e-5,
        visible=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/car.urdf")),
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
        pos=(0.0, 0.0, 0.0),
        joint_pos={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 3.141},
        joint_vel={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 0.000},
    ),
    soft_joint_pos_limit_factor=1.0,
    collision_group=0,
    actuators={
        "prismatic": IdealPDActuatorCfg(
            joint_names_expr=["dummy_prismatic.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_prismatic.*": 0},
            damping={"dummy_prismatic.*": 1000},
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": IdealPDActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_revolute.*": 0},
            damping={"dummy_revolute.*": 1000},
            friction={"dummy_revolute.*": 0.0},
         ),
         #"passive": IdealPDActuatorCfg(
         #   joint_names_expr=["passive_prismatic_z_joint"],
         #   effort_limit=0.0,
         #   velocity_limit=10.0,
         #   stiffness={"passive_prismatic_z_joint": 0},
         #   damping={"passive_prismatic_z_joint": 0},
         #   friction={"passive_prismatic_z_joint": 0},
         #)
    },
    debug_vis=True,
)

SPHERE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        activate_contact_sensors=False,
        self_collision=False,
        convex_decompose_mesh=True,
        link_density=1e-5,
        visible=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/sphere.urdf")),
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
        pos=(0.0, 0.0, 0.343),
        joint_pos={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 3.141},
        joint_vel={"dummy_prismatic.*" : 0.000, "dummy_revolute.*" : 0.000},
    ),
    soft_joint_pos_limit_factor=1.0,
    collision_group=0,
    actuators={
        "prismatic": IdealPDActuatorCfg(
            joint_names_expr=["dummy_prismatic.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_prismatic.*": 0},
            damping={"dummy_prismatic.*": 1000},
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": IdealPDActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_revolute.*": 0},
            damping={"dummy_revolute.*": 1000},
            friction={"dummy_revolute.*": 0.0},
         ),
         #"passive": IdealPDActuatorCfg(
         #   joint_names_expr=["passive_prismatic_z_joint"],
         #   effort_limit=0.0,
         #   velocity_limit=10.0,
         #   stiffness={"passive_prismatic_z_joint": 0},
         #   damping={"passive_prismatic_z_joint": 0},
         #   friction={"passive_prismatic_z_joint": 0},
         #)
    },
    debug_vis=True,
)


BUMPYBOT_POSE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        activate_contact_sensors=True,
        self_collision=False,
        convex_decompose_mesh=False,
        link_density=0.0,
        visible=True,
        asset_path=os.path.abspath(os.path.join(EXT_DIR, "resources/hcrl_robots/bumpybot/bumpybot.urdf")),
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
        pos=(0.0, 0.0, 0.543), #0.343
        joint_pos={".*" : 0.000},
        joint_vel={".*" : 0.000},
    ),
    soft_joint_pos_limit_factor=1.0,
    collision_group=0,
    actuators={
        "prismatic": IdealPDActuatorCfg(
            joint_names_expr=["dummy_prismatic_x_joint","dummy_prismatic_y_joint"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_prismatic.*": 10.0},
            damping={"dummy_prismatic.*": 0.0},
            friction={"dummy_prismatic.*": 0.0},
        ),
         "revolute": IdealPDActuatorCfg(
            joint_names_expr=["dummy_revolute.*"],
            effort_limit=1000.0,
            velocity_limit=10.0,
            stiffness={"dummy_revolute.*": 10.0},
            damping={"dummy_revolute.*": 0.0},
            friction={"dummy_revolute.*": 0.0},
         ),
         #"passive": IdealPDActuatorCfg(
         #   joint_names_expr=["passive_prismatic_z_joint"],
         #   effort_limit=0.0,
         #   velocity_limit=10.0,
         #   stiffness={"passive_prismatic_z_joint": 0},
         #   damping={"passive_prismatic_z_joint": 0},
         #   friction={"passive_prismatic_z_joint": 0},
         #)
    },
    debug_vis=True,
)



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
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
)

CUBE0_CFG = RigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        #size=(0.5, 0.75, 0.5),
        size=(1.0, 1.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4), opacity=1),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.5, -3.25, 0.25)),
)

CUBE1_CFG = RigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        #size=(1.0, 0.75, 0.5),
        size=(2.0, 1.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4), opacity=1),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -0.75, 0.25)),
)

CUBE2_CFG = RigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        #size=(1.0, 0.75, 0.5),
        size=(2.0, 2.0, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4), opacity=1),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 1.0, 0.25)),
)

CUBE3_CFG = RigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        #size=(1.0, 0.75, 0.5),
        size=(1.5, 1.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4), opacity=1),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
)