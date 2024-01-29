"""Package containing task implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
HCRL_ORBIT_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

HCRL_ORBIT_METADATA = toml.load(os.path.join(HCRL_ORBIT_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = HCRL_ORBIT_METADATA["package"]["version"]

##
# Register Gym environments.
##

from omni.isaac.orbit_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
#_BLACKLIST_PKGS = ["utils"]
_BLACKLIST_PKGS = []
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)