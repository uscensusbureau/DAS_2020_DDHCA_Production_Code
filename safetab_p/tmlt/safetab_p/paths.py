"""Paths used by SafeTab-P."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import pkgutil
from pathlib import Path
from typing import Dict

from tmlt.common.configuration import Config
from tmlt.safetab_utils.paths import INPUT_FILES_SAFETAB_P

RESOURCES_PACKAGE_NAME = "resources"
"""The name of the directory containing resources.

This is used by pkgutil.
"""

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), RESOURCES_PACKAGE_NAME)
)
"""Path to directory containing resources for SafeTab-P."""

INPUT_CONFIG_DIR = os.path.join(RESOURCES_DIR, "config/input")
"""Directory containing initial configs for input files."""

ALT_INPUT_CONFIG_DIR_SAFETAB_P = "/tmp/safetab_p_input_configs"
ALT_OUTPUT_CONFIG_DIR_SAFETAB_P = "/tmp/safetab_p_output_configs"
"""The config directories to use for spark-compatible version of SafeTab-P.

Config files are copied to this directory from safetab_p resources. They cannot be used
directly because SafeTab-P resources may be zipped.
"""

OUTPUT_CONFIG_FILES = ["t1", "t2"]
"""The names of the expected output files (without file extensions)."""


def setup_input_config_dir():
    """Copy INPUT_CONFIG_DIR contents to temp directory ALT_INPUT_CONFIG_DIR_SAFETAB_P.

    NOTE: This setup is required to ensure zip-compatibility of Safetab.
    In particular, configs in resources directory of Safetab can not read
    when Safetab is distributed (and invoked) as zip archive (See Issue #331)
    """
    os.makedirs(ALT_INPUT_CONFIG_DIR_SAFETAB_P, exist_ok=True)
    for cfg_file in set(INPUT_FILES_SAFETAB_P):
        json_filename = os.path.splitext(cfg_file)[0] + ".json"
        json_file = Path(os.path.join(ALT_INPUT_CONFIG_DIR_SAFETAB_P, json_filename))
        json_file.touch(exist_ok=True)
        json_file.write_bytes(
            pkgutil.get_data(
                "tmlt.safetab_p", os.path.join("resources/config/input", json_filename)
            )
        )


def setup_safetab_p_output_config_dir():
    """Copy output contents to temp dir ALT_OUTPUT_CONFIG_DIR_SAFETAB_P.

    NOTE: This setup is required to ensure zip-compatibility of Safetab.
    In particular, configs in resources directory of Safetab can not read
    when Safetab is distributed (and invoked) as zip archive (See Issue #331)
    """
    os.makedirs(ALT_OUTPUT_CONFIG_DIR_SAFETAB_P, exist_ok=True)
    for cfg_file in OUTPUT_CONFIG_FILES:
        json_filename = cfg_file + ".json"
        json_file = Path(os.path.join(ALT_OUTPUT_CONFIG_DIR_SAFETAB_P, json_filename))
        json_file.touch(exist_ok=True)
        json_file.write_bytes(
            pkgutil.get_data(
                "tmlt.safetab_p", os.path.join("resources/config/output", json_filename)
            )
        )


def get_safetab_p_output_configs() -> Dict[str, Config]:
    """Returns a dictionary mapping output subdirectories to configs.

    The configs in the dict describe the output file schemas for each subdirectory.
    """
    configs = {}
    for cfg_file in OUTPUT_CONFIG_FILES:
        json_filename = f"{cfg_file}.json"
        configs[cfg_file] = Config.load_json(
            os.path.join(ALT_OUTPUT_CONFIG_DIR_SAFETAB_P, json_filename)
        )
    return configs
