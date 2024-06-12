from __future__ import annotations
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac.lab.hcrl import EXT_DIR

##
# Configuration
##


HOSPITAL_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    )
)

ROOM_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
    )
)

DEFAULT_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
    )
)
