"""Tests SafeTab-P config validation."""

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

import copy
import json
import os
import re
from typing import Dict, List

from parameterized import parameterized

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.paths import RESOURCES_DIR
from tmlt.safetab_utils.config_validation import validate_config_values


class TestConfigValidation(PySparkTest):
    """Tests invalid SafeTab-P config fails config validation."""

    def setUp(self):
        """Set up test."""
        self.parameters_path = os.path.join(RESOURCES_DIR, "toy_dataset/input_dir_zcdp")
        with open(os.path.join(self.parameters_path, "config.json"), "r") as f:
            self.temp_config_json = json.load(f)

    @parameterized.expand(
        [
            (
                {"privacy_defn": "pure"},
                r"Invalid config: Supported privacy definitions are Rho zCDP \(zcdp\)"
                r" and Pure DP \(puredp\).",
            ),
            (
                {"max_race_codes": "random"},
                "Invalid config: expected 'max_race_codes' to have int value"
                " between 1 and 8.",
            ),
            (
                {"max_race_codes": 9},
                "expected 'max_race_codes' to have int value between 1 and 8.",
            ),
            (
                {"allow_negative_counts": "xyz"},
                "Supported value for 'allow_negative_counts' are true and false.",
            ),
            ({"run_us": "Yes"}, "Supported value for 'run_us' are true and false."),
            ({"run_pr": 10}, "Supported value for 'run_pr' are true and false."),
            (
                {"zero_suppression_chance": -0.5},
                "Invalid config: expected zero_suppression_chance to have a float"
                " value between 0 and 1, but got -0.5.",
            ),
            (
                {"zero_suppression_chance": 1.0},
                "Invalid config: expected zero_suppression_chance to have a float"
                " value between 0 and 1, but got 1.0.",
            ),
            (
                {"zero_suppression_chance": 1.5},
                "Invalid config: expected zero_suppression_chance to have a float"
                " value between 0 and 1, but got 1.5.",
            ),
        ]
    )
    def test_invalid_config(self, overwrite_config: Dict, error_msg_regex: str):
        """validate_config_values errors for invalid common config keys.

        Args:
            overwrite_config: JSON data to be validated, as a Python dict
            error_msg_regex: A regular expression to check the error message against
        """
        invalid_config = copy.deepcopy(self.temp_config_json)
        for key, value in overwrite_config.items():
            invalid_config[key] = value

        with self.assertRaisesRegex(ValueError, error_msg_regex):
            validate_config_values(invalid_config, "safetab-p", ["US", "PR"])

    @parameterized.expand(
        [
            (
                {},
                ["US", "PR"],
                "missing required keys {allow_negative_counts, max_race_codes,"
                " privacy_budget_p_level_1_aiannh, privacy_budget_p_level_1_county,"
                " privacy_budget_p_level_1_place,"
                " privacy_budget_p_level_1_pr_county,"
                " privacy_budget_p_level_1_pr_place,"
                " privacy_budget_p_level_1_pr_state,"
                " privacy_budget_p_level_1_pr_tract,"
                " privacy_budget_p_level_1_state, privacy_budget_p_level_1_tract,"
                " privacy_budget_p_level_1_usa, privacy_budget_p_level_2_aiannh,"
                " privacy_budget_p_level_2_county, privacy_budget_p_level_2_place,"
                " privacy_budget_p_level_2_pr_county,"
                " privacy_budget_p_level_2_pr_place,"
                " privacy_budget_p_level_2_pr_state,"
                " privacy_budget_p_level_2_pr_tract,"
                " privacy_budget_p_level_2_state, privacy_budget_p_level_2_tract,"
                " privacy_budget_p_level_2_usa, privacy_budget_p_stage_1_fraction,"
                " privacy_defn, reader, run_pr, run_us, state_filter_us,"
                " thresholds_p}.",
            ),
            (
                {"privacy_budget_p_stage_1_fraction": 0},
                ["US", "PR"],
                "'privacy_budget_p_stage_1_fraction' must be between 0 and 1.",
            ),
            (
                {"privacy_budget_p_stage_1_fraction": -1},
                ["US", "PR"],
                "'privacy_budget_p_stage_1_fraction' must be between 0 and 1.",
            ),
            (
                {
                    "run_us": True,
                    "run_pr": True,
                    "thresholds_p": {
                        "(USA, 1)": [5000, 20000, 150000],
                        "(PR-COUNTY, 2)": [1000, 5000, 20000],
                        "(PR-PLACE, 1)": [1000, 5000, 20000],
                        "(PR-PLACE, 2)": [1000, 5000, 20000],
                    },
                },
                # execute_plan_p_analytics calls US/PR and not both -
                # this is called in accuracy report as well
                ["PR"],
                "missing required keys in 'thresholds_p' {(PR-COUNTY, 1),"
                " (PR-STATE, 1), (PR-STATE, 2), (PR-TRACT, 1), (PR-TRACT, 2)}."
                " Ensure thresholds for each combination of geography level and"
                " iteration level for safetab-p PR run is specified.",
            ),
            (
                {
                    "run_us": True,
                    "run_pr": True,
                    "thresholds_p": {
                        "(USA, 1)": [5000, 20000, 150000],
                        "(USA, 2)": [500, 1000, 7000],
                        "(PLACE, 2)": [1000, 5000, 20000],
                        "(AIANNH, 1)": [5000, 20000, 150000],
                        "(AIANNH, 2)": [1000, 5000, 20000],
                        "(PR-STATE, 2)": [500, 1000, 7000],
                    },
                },
                # execute_plan_p_analytics calls US/PR and not both -
                # this is called in accuracy report as well
                ["US"],
                "missing required keys in 'thresholds_p' {(COUNTY, 1), (COUNTY, 2),"
                " (PLACE, 1), (STATE, 1), (STATE, 2), (TRACT, 1), (TRACT, 2)}."
                " Ensure thresholds for each combination of geography level and"
                " iteration level for safetab-p US run is specified.",
            ),
            (
                {
                    "thresholds_p": {
                        "(USA, 1)": [5000, 20000],
                        "(USA, 2)": [500, 1000, 7000],
                        "(STATE, 1)": [5000, 20000, 150000],
                        "(STATE, 2)": [500, 1000, 7000],
                        "(COUNTY, 1)": [5000, 20000, 150000],
                        "(COUNTY, 2)": [1000, 5000, 20000],
                        "(TRACT, 1)": [5000, 20000, 150000],
                        "(TRACT, 2)": [1000, 5000, 20000],
                        "(PLACE, 1)": [5000, 20000, 150000],
                        "(PLACE, 2)": [1000, 5000, 20000],
                        "(AIANNH, 1)": [5000, 20000, 150000],
                        "(AIANNH, 2)": [1000, 5000, 20000],
                        "(PR-STATE, 1)": [5000, 20000, 150000],
                        "(PR-STATE, 2)": [500, 1000, 7000],
                        "(PR-COUNTY, 1)": [5000, 20000, 150000],
                        "(PR-COUNTY, 2)": [1000, 5000, 20000],
                        "(PR-TRACT, 1)": [1000, 5000, 20000],
                        "(PR-TRACT, 2)": [1000, 5000, 20000],
                        "(PR-PLACE, 1)": [1000, 5000, 20000],
                        "(PR-PLACE, 2)": [1000, 5000, 20000],
                    }
                },
                ["US", "PR"],
                "At least 3 non-decreasing thresholds for (USA, 1) should be"
                " specified in 'thresholds_p'.",
            ),
            (
                {
                    "thresholds_p": {
                        "(USA, 1)": [20000, 20000, 150000],
                        "(USA, 2)": [500, 1000, 7000],
                        "(STATE, 1)": [5000, 20000, 150000],
                        "(STATE, 2)": [500, 1000, 7000],
                        "(COUNTY, 1)": [5000, 20000, 150000],
                        "(COUNTY, 2)": [1000, 5000, 20000],
                        "(TRACT, 1)": [5000, 20000, 150000],
                        "(TRACT, 2)": [1000, 5000, 20000],
                        "(PLACE, 1)": [5000, 20000, 150000],
                        "(PLACE, 2)": [1000, 5000, 20000],
                        "(AIANNH, 1)": [5000, 20000, 150000],
                        "(AIANNH, 2)": [1000, 5000, 20000],
                        "(PR-STATE, 1)": [20000, 5000, 150000],
                        "(PR-STATE, 2)": [500, 1000, 7000],
                        "(PR-COUNTY, 1)": [5000, 20000, 150000],
                        "(PR-COUNTY, 2)": [1000, 5000, 20000],
                        "(PR-TRACT, 1)": [1000, 5000, 20000],
                        "(PR-TRACT, 2)": [1000, 5000, 20000],
                        "(PR-PLACE, 1)": [1000, 5000, 20000],
                        "(PR-PLACE, 2)": [1000, 5000, 20000],
                    }
                },
                ["US", "PR"],
                "At least 3 non-decreasing thresholds for (PR-STATE, 1) should be"
                " specified in 'thresholds_p'.",
            ),
        ]
    )
    def test_invalid_program_specific_config(
        self,
        overwrite_config: Dict,
        us_or_puerto_rico_values: List[str],
        error_msg_regex: str,
    ):
        """validate_config_values errors for invalid config keys specific to safetab-p.

        Args:
            overwrite_config: JSON data to be validated, as a Python dict
            us_or_puerto_rico_values: Speficies run - US or PR or both
            error_msg_regex: A regular expression to check the error message against
        """
        error_msg_regex = re.escape(error_msg_regex)
        if overwrite_config:
            invalid_config = copy.deepcopy(self.temp_config_json)
            for key, value in overwrite_config.items():
                invalid_config[key] = value
        else:
            invalid_config = {}

        with self.assertRaisesRegex(ValueError, error_msg_regex):
            validate_config_values(
                invalid_config, "safetab-p", us_or_puerto_rico_values
            )
