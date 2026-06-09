# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Package containing asset and sensor configurations for IRobot_wl."""

import os

import toml

IROBOT_WL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

IROBOT_WL_DATA_DIR = os.path.join(IROBOT_WL_EXT_DIR, "data")
"""Path to the extension data directory."""

IROBOT_WL_METADATA = toml.load(os.path.join(IROBOT_WL_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

__version__ = IROBOT_WL_METADATA["package"]["version"]
