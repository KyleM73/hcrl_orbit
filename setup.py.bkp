"""Installation script for the 'omni.isaac.orbit_tasks' python package."""

import itertools
import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    "protobuf>=3.20.2",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "rsl_rl": ["rsl_rl@git+https://github.com/KyleM73/rsl_rl.git"],
}
# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))


# Installation operation
setup(
    name="hcrl_orbit",
    author="Kyle Morgenstein",
    maintainer="Kyle Morgenstein",
    maintainer_email="kylem@utexas.edu",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["hcrl_orbit"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.0-hotfix.1",
    ],
    zip_safe=False,
)
