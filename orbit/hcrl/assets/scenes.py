from __future__ import annotations
import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from orbit.hcrl import EXT_DIR

##
# Configuration
##

HOSPITAL_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    )
)