"""Module for config validation prior to private algorithm execution."""

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

import itertools
from fractions import Fraction
from typing import Any, Dict, List

from tmlt.safetab_utils.regions import REGION_TYPES

CONFIG_PARAMS_P = [
    "max_race_codes",
    "privacy_budget_p_level_1_usa",
    "privacy_budget_p_level_2_usa",
    "privacy_budget_p_level_1_state",
    "privacy_budget_p_level_2_state",
    "privacy_budget_p_level_1_county",
    "privacy_budget_p_level_2_county",
    "privacy_budget_p_level_1_tract",
    "privacy_budget_p_level_2_tract",
    "privacy_budget_p_level_1_place",
    "privacy_budget_p_level_2_place",
    "privacy_budget_p_level_1_aiannh",
    "privacy_budget_p_level_2_aiannh",
    "privacy_budget_p_level_1_pr_state",
    "privacy_budget_p_level_2_pr_state",
    "privacy_budget_p_level_1_pr_county",
    "privacy_budget_p_level_2_pr_county",
    "privacy_budget_p_level_1_pr_tract",
    "privacy_budget_p_level_2_pr_tract",
    "privacy_budget_p_level_1_pr_place",
    "privacy_budget_p_level_2_pr_place",
    "privacy_budget_p_stage_1_fraction",
    "privacy_budget_p_stage_1_fraction",
    "thresholds_p",
    "allow_negative_counts",
    "run_us",
    "run_pr",
    "reader",
    "state_filter_us",
    "privacy_defn",
]
"""`config.json` params that can change the behavior of safetab-p."""

CONFIG_PARAMS_H = [
    "max_race_codes",
    "privacy_budget_h_t3_level_1_usa",
    "privacy_budget_h_t3_level_2_usa",
    "privacy_budget_h_t3_level_1_state",
    "privacy_budget_h_t3_level_2_state",
    "privacy_budget_h_t3_level_1_county",
    "privacy_budget_h_t3_level_2_county",
    "privacy_budget_h_t3_level_1_tract",
    "privacy_budget_h_t3_level_2_tract",
    "privacy_budget_h_t3_level_1_place",
    "privacy_budget_h_t3_level_2_place",
    "privacy_budget_h_t3_level_1_aiannh",
    "privacy_budget_h_t3_level_2_aiannh",
    "privacy_budget_h_t3_level_1_pr_state",
    "privacy_budget_h_t3_level_2_pr_state",
    "privacy_budget_h_t3_level_1_pr_county",
    "privacy_budget_h_t3_level_2_pr_county",
    "privacy_budget_h_t3_level_1_pr_tract",
    "privacy_budget_h_t3_level_2_pr_tract",
    "privacy_budget_h_t3_level_1_pr_place",
    "privacy_budget_h_t3_level_2_pr_place",
    "privacy_budget_h_t4_level_1_usa",
    "privacy_budget_h_t4_level_2_usa",
    "privacy_budget_h_t4_level_1_state",
    "privacy_budget_h_t4_level_2_state",
    "privacy_budget_h_t4_level_1_county",
    "privacy_budget_h_t4_level_2_county",
    "privacy_budget_h_t4_level_1_tract",
    "privacy_budget_h_t4_level_2_tract",
    "privacy_budget_h_t4_level_1_place",
    "privacy_budget_h_t4_level_2_place",
    "privacy_budget_h_t4_level_1_aiannh",
    "privacy_budget_h_t4_level_2_aiannh",
    "privacy_budget_h_t4_level_1_pr_state",
    "privacy_budget_h_t4_level_2_pr_state",
    "privacy_budget_h_t4_level_1_pr_county",
    "privacy_budget_h_t4_level_2_pr_county",
    "privacy_budget_h_t4_level_1_pr_tract",
    "privacy_budget_h_t4_level_2_pr_tract",
    "privacy_budget_h_t4_level_1_pr_place",
    "privacy_budget_h_t4_level_2_pr_place",
    "thresholds_h_t3",
    "thresholds_h_t4",
    "allow_negative_counts",
    "run_us",
    "run_pr",
    "reader",
    "state_filter_us",
    "privacy_defn",
]
"""`config.json` params that can change the behavior of safetab-h."""

RACE_CODE_VALID_COUNTS = range(1, 9)

MIN_THRESHOLDS = 3


def validate_config_values(
    config: Dict[str, Any], program: str, us_or_puerto_rico_values: List[str]
):
    """Ensure sensible values in input config for private algorithm execution.

    Checks that config values in the given config dictionary are both the
    appropriate types and that they have allowed values.

    STATE_FILTER_FLAG and READER_FLAG are checked separately as these are
    required for standalone validate mode.

    Args:
        config: A configuration dictionary to validate. It is assumed that the keys
            in this dictionary have been validated elsewhere.
        program: Allowed options are 'safetab-p' and 'safetab-h'.
        us_or_puerto_rico_values: Decides if this is "US"/
            "PR"/both run.

    Raises:
        ValueError: When any of the config values are invalid.
    """

    if program not in {"safetab-p", "safetab-h"}:
        raise ValueError(
            f"'program' must be 'safetab-p' or 'safetab-h', not '{program}'."
        )

    required_config_keys = (
        CONFIG_PARAMS_P if program == "safetab-p" else CONFIG_PARAMS_H
    )

    missing_keys = sorted(set(required_config_keys) - set(config))
    if missing_keys:
        missing_keys_str = "{" + ", ".join(missing_keys) + "}"
        raise ValueError(f"Invalid config: missing required keys {missing_keys_str}.")

    if not us_or_puerto_rico_values or any(
        item not in ["US", "PR"] for item in us_or_puerto_rico_values
    ):
        raise ValueError(
            "'us_or_puerto_rico_values' must be 'US' and/or 'PR', not"
            f" '{us_or_puerto_rico_values}'."
        )

    if config["privacy_defn"] not in ["zcdp", "puredp"]:
        raise ValueError(
            "Invalid config: Supported privacy definitions are Rho zCDP (zcdp) and Pure"
            " DP (puredp)."
        )

    max_race_code_value = config["max_race_codes"]
    if (
        not isinstance(max_race_code_value, int)
        or max_race_code_value not in RACE_CODE_VALID_COUNTS
    ):
        raise ValueError(
            "Invalid config: expected 'max_race_codes' to have int value between"
            f" {min(RACE_CODE_VALID_COUNTS)} and {max(RACE_CODE_VALID_COUNTS)}."
        )

    for key in ["allow_negative_counts", "run_us", "run_pr"]:
        if not isinstance(config[key], bool):
            raise ValueError(
                f"Invalid config: Supported value for '{key}' are true and false."
            )

    if not us_or_puerto_rico_values:
        # Never raised as `run_us`, `run_pr` check happens early
        raise ValueError("At least one of 'US', 'PR' must be specified.")

    if program == "safetab-p":
        stage_1_budget_fraction = Fraction(config["privacy_budget_p_stage_1_fraction"])
        if not 0 < stage_1_budget_fraction < 1:
            raise ValueError(
                "Invalid config: 'privacy_budget_p_stage_1_fraction' must be between 0"
                " and 1."
            )

        if "zero_suppression_chance" in config:
            if (
                config["zero_suppression_chance"] < 0.0
                or config["zero_suppression_chance"] >= 1
            ):
                raise ValueError(
                    "Invalid config: expected zero_suppression_chance to have a"
                    " float value between 0 and 1, but got"
                    f" {config['zero_suppression_chance']}."
                )

        for us_or_puerto_rico in us_or_puerto_rico_values:
            required_threshold_p_keys = []
            for region_type, iteration_level in itertools.product(
                REGION_TYPES[us_or_puerto_rico], ["1", "2"]
            ):
                required_threshold_p_keys.append(f"({region_type}, {iteration_level})")

            missing_threshold_keys = sorted(
                set(required_threshold_p_keys) - set(config["thresholds_p"])
            )
            if missing_threshold_keys:
                missing_threshold_keys_str = (
                    "{" + ", ".join(missing_threshold_keys) + "}"
                )
                raise ValueError(
                    "Invalid config: missing required keys in 'thresholds_p'"
                    f" {missing_threshold_keys_str}. Ensure thresholds for each"
                    " combination of geography level and iteration level for safetab-p"
                    f" {us_or_puerto_rico} run is specified."
                )
            for pop_group, thresholds_list in config["thresholds_p"].items():
                if len(thresholds_list) < MIN_THRESHOLDS or not all(
                    x <= y for x, y in zip(thresholds_list, thresholds_list[1:])
                ):
                    raise ValueError(
                        f"Invalid config: At least {MIN_THRESHOLDS} non-decreasing"
                        f" thresholds for {pop_group} should be specified in"
                        " 'thresholds_p'."
                    )
    else:
        for key, budget in config.items():
            if not key.startswith("privacy_budget_h_t"):
                continue
            if isinstance(budget, int):
                budget = float(budget)
            if not isinstance(budget, float):
                raise ValueError(
                    f"Invalid config: expected '{key}' key "
                    f"to have float value, not {type(budget).__name__}."
                )
            if budget < 0:
                raise ValueError(f"Invalid config: '{key}' must be non-negative.")

        for us_or_puerto_rico in us_or_puerto_rico_values:
            required_threshold_h_keys = []
            for region_type, iteration_level in itertools.product(
                REGION_TYPES[us_or_puerto_rico], ["1", "2"]
            ):
                required_threshold_h_keys.append(f"({region_type}, {iteration_level})")

            for table in ["t3", "t4"]:
                missing_threshold_keys = sorted(
                    set(required_threshold_h_keys)
                    - set(config[f"thresholds_h_{table}"])
                )

                if missing_threshold_keys:
                    missing_threshold_keys_str = (
                        "{" + ", ".join(missing_threshold_keys) + "}"
                    )
                    raise ValueError(
                        "Invalid config: missing required keys in 'thresholds_h'"
                        f" {missing_threshold_keys_str}. Ensure thresholds for each"
                        " combination of geography level and iteration level for"
                        f" safetab-h {us_or_puerto_rico} run is specified."
                    )

            for pop_group, thresholds_list in config["thresholds_h_t3"].items():
                if len(thresholds_list) < 3 or not all(
                    x <= y for x, y in zip(thresholds_list, thresholds_list[1:])
                ):
                    raise ValueError(
                        "Invalid config: At least three non-decreasing thresholds for"
                        f" t3 {pop_group} should be specified in 'thresholds_h_t3'."
                        f" Thresholds: {thresholds_list}"
                    )

            for pop_group, thresholds_list in config["thresholds_h_t4"].items():
                if len(thresholds_list) != 1:
                    raise ValueError(
                        "Invalid config: One non-decreasing threshold for"
                        f" t4 {pop_group} should be specified in 'thresholds_h_t4'."
                    )
