"""System tests for SafeTab, making sure that the algorithm has the correct output."""

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

# pylint: disable=protected-access, no-member
import json
import os
import shutil
import tempfile
from typing import Callable, Dict

import numpy as np
import pandas as pd
from nose.plugins.attrib import attr
from parameterized import param, parameterized, parameterized_class

from tmlt.common.io_helpers import multi_read_csv
from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.paths import INPUT_CONFIG_DIR, RESOURCES_DIR
from tmlt.safetab_p.safetab_p_analytics import (
    execute_plan_p_analytics,
    run_plan_p_analytics,
)
from tmlt.safetab_p.target_counts_p import (
    WORKLOAD_SIZE_TO_STAT_LEVEL,
    create_ground_truth_p,
)
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    get_augmented_df,
    safetab_input_reader,
    validate_directory_single_config,
)

query_privacy_budgets = [
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
]

INFINITE_PRIVACY_BUDGET_P = {key: float("inf") for key in query_privacy_budgets}

TEN_PRIVACY_BUDGET_P = {key: 10.0 for key in query_privacy_budgets}

MANUAL_THRESHOLDS = {
    "(USA, 1)": list(range(3)),
    "(USA, 2)": list(range(3)),
    "(STATE, 1)": list(range(3)),
    "(STATE, 2)": list(range(3)),
    "(COUNTY, 1)": list(range(3)),
    "(COUNTY, 2)": list(range(3)),
    "(TRACT, 1)": list(range(3)),
    "(TRACT, 2)": list(range(3)),
    "(PLACE, 1)": list(range(3)),
    "(PLACE, 2)": list(range(3)),
    "(AIANNH, 1)": list(range(3)),
    "(AIANNH, 2)": list(range(3)),
    "(PR-STATE, 1)": list(range(3)),
    "(PR-STATE, 2)": list(range(3)),
    "(PR-COUNTY, 1)": list(range(3)),
    "(PR-COUNTY, 2)": list(range(3)),
    "(PR-TRACT, 1)": list(range(3)),
    "(PR-TRACT, 2)": list(range(3)),
    "(PR-PLACE, 1)": list(range(3)),
    "(PR-PLACE, 2)": list(range(3)),
}


@parameterized_class(
    [
        {
            "name": "SafeTab-P US Analytics",
            "us_or_puerto_rico": "US",
            "parameters_input_dir": "input_dir_puredp",
        },
        {
            "name": "SafeTab-P PR Analytics",
            "us_or_puerto_rico": "PR",
            "parameters_input_dir": "input_dir_puredp",
        },
        {
            "name": "SafeTab-P US Analytics",
            "us_or_puerto_rico": "US",
            "parameters_input_dir": "input_dir_zcdp",
        },
        {
            "name": "SafeTab-P PR Analytics",
            "us_or_puerto_rico": "PR",
            "parameters_input_dir": "input_dir_zcdp",
        },
    ]
)
class TestAlgorithmWithTargetOutput(PySparkTest):
    """Test algorithm compared to target output."""

    name: str
    us_or_puetro_rico: str
    algorithms: Dict[str, Callable]
    parameters_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        print(self.name)  # Hint to determine what algorithm is being tested.
        self.output_files = ["t1", "t2"]
        # input_dir contains the parameters and CEF data files.
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        self.actual_dir = tempfile.TemporaryDirectory()
        self.expected_dir = tempfile.TemporaryDirectory()
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.config_dir = tempfile.TemporaryDirectory()
        with open(
            os.path.join(
                os.path.join(self.data_path, self.parameters_input_dir), "config.json"
            ),
            "r",
        ) as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            # This test runs either US or PR, never both
            if self.us_or_puerto_rico == "US" and validate_state_filter_us(
                config_json[STATE_FILTER_FLAG]
            ):
                state_filter = config_json[STATE_FILTER_FLAG]
            else:
                state_filter = ["72"]
            validate_input(  # Create configs used to run SafeTab.
                parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
                input_data_configs_path=INPUT_CONFIG_DIR,
                output_path=self.config_dir.name,
                program="safetab-p",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=self.data_path,
                    state_filter=state_filter,
                    program="safetab-p",
                ),
                state_filter=state_filter,
            )

    def test_infinite_eps(self):
        """SafeTab-P has zero error with infinite budget."""

        execute_plan_p_analytics(
            parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
            data_path=self.data_path,
            output_path=self.actual_dir.name,
            config_path=self.config_dir.name,
            overwrite_config={
                **INFINITE_PRIVACY_BUDGET_P,
                "thresholds_p": MANUAL_THRESHOLDS,
            },
            us_or_puerto_rico=self.us_or_puerto_rico,
        )
        create_ground_truth_p(
            parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
            data_path=self.data_path,
            output_path=self.expected_dir.name,
            config_path=self.config_dir.name,
            # Overwriting the privacy budget is necessary to tabulate all geo/iteration
            # levels, including those with budget set to 0 in the default config.json.
            overwrite_config={**INFINITE_PRIVACY_BUDGET_P},
            us_or_puerto_rico=self.us_or_puerto_rico,
        )

        for output_file in self.output_files:
            print(f"Checking {output_file}")
            augmented_df = get_augmented_df(
                output_file, self.actual_dir.name, self.expected_dir.name
            )
            self.assert_frame_equal_with_sort(
                augmented_df.rename(columns={"NOISY": "COUNT"}).drop(
                    columns="GROUND_TRUTH"
                ),
                augmented_df.rename(columns={"GROUND_TRUTH": "COUNT"}).drop(
                    columns="NOISY"
                ),
                list(set(augmented_df.columns) - {"NOISY", "GROUND_TRUTH"}),
            )

    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_infinite_eps_using_safetab_output(self):
        """SafeTab-P has zero error with infinite budget.

        Use SafeTab-P output to determine workloads for the ground truth.
        """
        execute_plan_p_analytics(
            parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
            data_path=self.data_path,
            output_path=self.actual_dir.name,
            config_path=self.config_dir.name,
            overwrite_config={
                **INFINITE_PRIVACY_BUDGET_P,
                "thresholds_p": MANUAL_THRESHOLDS,
            },
            us_or_puerto_rico=self.us_or_puerto_rico,
        )
        create_ground_truth_p(
            parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
            data_path=self.data_path,
            output_path=self.expected_dir.name,
            config_path=self.config_dir.name,
            safetab_output_paths=[self.actual_dir.name],
            # Overwriting the privacy budget is necessary to tabulate all geo/iteration
            # levels, including those with budget set to 0 in the default config.json.
            overwrite_config=INFINITE_PRIVACY_BUDGET_P,
            us_or_puerto_rico=self.us_or_puerto_rico,
        )

        t1_df = get_augmented_df("t1", self.actual_dir.name, self.expected_dir.name)
        t2_df = get_augmented_df("t2", self.actual_dir.name, self.expected_dir.name)
        pd.testing.assert_series_equal(
            t1_df["NOISY"], t1_df["GROUND_TRUTH"], check_names=False
        )
        pd.testing.assert_series_equal(
            t2_df["NOISY"], t2_df["GROUND_TRUTH"], check_names=False
        )


# pylint: disable=line-too-long
# Note: The following input has a row to pass input validation but the tests will
# evaluate for state = 1
STATE_FILTERABLE_INPUT = """QAGE|QSEX|HOUSEHOLDER|TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|CENRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN
028|1|True|04|004|000004|0004|01|1010|Null|Null|Null|Null|Null|Null|Null|2020
028|1|True|72|002|000002|0002|01|1010|Null|Null|Null|Null|Null|Null|Null|2020
"""

SIMPLE_ETHNICITY_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|DETAILED_ONLY|COARSE_ONLY
3010|Central American|1|False|False
3011|Costa Rican|2|True|False"""

SIMPLE_RACE_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
2001|European alone|1|True|False|False
3001|Albanian alone|2|True|False|False"""

SIMPLE_RACE_AND_ETHNICITY_CODES = """RACE_ETH_CODE|RACE_ETH_NAME
1010|Albanian1
2020|Costa Rican1"""

SIMPLE_CODE_ITERATION_MAP = """ITERATION_CODE|RACE_ETH_CODE
2001|1010
3001|1010
3010|2020
3011|2020"""

SIMPLE_US_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
04|004|000004|0004|22801|0002
"""

# To simplify things, only the values in the state "01" and the US, "1", are used.
US_CHECKED_REGIONS = ["1", "01", "0122800", "01001", "01001000001", "0001"]
PR_CHECKED_REGIONS = ["72", "72001", "7222800", "72001000001", "0001"]

SIMPLE_PR_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
72|001|000001|0001|22800|0001
72|002|000002|0002|22801|0002
"""

SIMPLE_CONFIG = """{
  "max_race_codes": 8,
  "privacy_budget_p_level_1_usa": 1,
  "privacy_budget_p_level_2_usa": 1,
  "privacy_budget_p_level_1_state": 1,
  "privacy_budget_p_level_2_state": 1,
  "privacy_budget_p_level_1_county": 1,
  "privacy_budget_p_level_2_county": 1,
  "privacy_budget_p_level_1_tract": 1,
  "privacy_budget_p_level_2_tract": 1,
  "privacy_budget_p_level_1_place": 1,
  "privacy_budget_p_level_2_place": 1,
  "privacy_budget_p_level_1_aiannh": 1,
  "privacy_budget_p_level_2_aiannh": 1,
  "privacy_budget_p_level_1_pr_state": 1,
  "privacy_budget_p_level_2_pr_state": 1,
  "privacy_budget_p_level_1_pr_county": 1,
  "privacy_budget_p_level_2_pr_county": 1,
  "privacy_budget_p_level_1_pr_tract": 1,
  "privacy_budget_p_level_2_pr_tract": 1,
  "privacy_budget_p_level_1_pr_place": 1,
  "privacy_budget_p_level_2_pr_place": 1,
  "privacy_budget_p_stage_1_fraction": 0.1,
  "thresholds_p": {
      "(USA, 1)": [1000000, 2000000, 3000000],
      "(USA, 2)": [1000000, 2000000, 3000000],
      "(STATE, 1)": [1000000, 2000000, 3000000],
      "(STATE, 2)": [1000000, 2000000, 3000000],
      "(COUNTY, 1)": [1000000, 2000000, 3000000],
      "(COUNTY, 2)": [1000000, 2000000, 3000000],
      "(TRACT, 1)": [1000000, 2000000, 3000000],
      "(TRACT, 2)": [1000000, 2000000, 3000000],
      "(PLACE, 1)": [1000000, 2000000, 3000000],
      "(PLACE, 2)": [1000000, 2000000, 3000000],
      "(AIANNH, 1)": [1000000, 2000000, 3000000],
      "(AIANNH, 2)": [1000000, 2000000, 3000000],
      "(PR-STATE, 1)": [1000000, 2000000, 3000000],
      "(PR-STATE, 2)": [1000000, 2000000, 3000000],
      "(PR-COUNTY, 1)": [1000000, 2000000, 3000000],
      "(PR-COUNTY, 2)": [1000000, 2000000, 3000000],
      "(PR-TRACT, 1)": [1000000, 2000000, 3000000],
      "(PR-TRACT, 2)": [1000000, 2000000, 3000000],
      "(PR-PLACE, 1)": [1000000, 2000000, 3000000],
      "(PR-PLACE, 2)": [1000000, 2000000, 3000000]
  },
  "zero_suppression_chance": 0.99999,
  "allow_negative_counts": true,
  "run_us": true,
  "run_pr": false,
  "reader": "csv",
  "state_filter_us": ["01", "04"],
  "privacy_defn": "puredp"
}
"""

CONFIG_NO_SUPPRESSION = """{
  "max_race_codes": 8,
  "privacy_budget_p_level_1_usa": 1,
  "privacy_budget_p_level_2_usa": 1,
  "privacy_budget_p_level_1_state": 1,
  "privacy_budget_p_level_2_state": 1,
  "privacy_budget_p_level_1_county": 1,
  "privacy_budget_p_level_2_county": 1,
  "privacy_budget_p_level_1_tract": 1,
  "privacy_budget_p_level_2_tract": 1,
  "privacy_budget_p_level_1_place": 1,
  "privacy_budget_p_level_2_place": 1,
  "privacy_budget_p_level_1_aiannh": 1,
  "privacy_budget_p_level_2_aiannh": 1,
  "privacy_budget_p_level_1_pr_state": 1,
  "privacy_budget_p_level_2_pr_state": 1,
  "privacy_budget_p_level_1_pr_county": 1,
  "privacy_budget_p_level_2_pr_county": 1,
  "privacy_budget_p_level_1_pr_tract": 1,
  "privacy_budget_p_level_2_pr_tract": 1,
  "privacy_budget_p_level_1_pr_place": 1,
  "privacy_budget_p_level_2_pr_place": 1,
  "privacy_budget_p_stage_1_fraction": 0.1,
  "thresholds_p": {
      "(USA, 1)": [1000000, 2000000, 3000000],
      "(USA, 2)": [1000000, 2000000, 3000000],
      "(STATE, 1)": [1000000, 2000000, 3000000],
      "(STATE, 2)": [1000000, 2000000, 3000000],
      "(COUNTY, 1)": [1000000, 2000000, 3000000],
      "(COUNTY, 2)": [1000000, 2000000, 3000000],
      "(TRACT, 1)": [1000000, 2000000, 3000000],
      "(TRACT, 2)": [1000000, 2000000, 3000000],
      "(PLACE, 1)": [1000000, 2000000, 3000000],
      "(PLACE, 2)": [1000000, 2000000, 3000000],
      "(AIANNH, 1)": [1000000, 2000000, 3000000],
      "(AIANNH, 2)": [1000000, 2000000, 3000000],
      "(PR-STATE, 1)": [1000000, 2000000, 3000000],
      "(PR-STATE, 2)": [1000000, 2000000, 3000000],
      "(PR-COUNTY, 1)": [1000000, 2000000, 3000000],
      "(PR-COUNTY, 2)": [1000000, 2000000, 3000000],
      "(PR-TRACT, 1)": [1000000, 2000000, 3000000],
      "(PR-TRACT, 2)": [1000000, 2000000, 3000000],
      "(PR-PLACE, 1)": [1000000, 2000000, 3000000],
      "(PR-PLACE, 2)": [1000000, 2000000, 3000000]
  },
  "allow_negative_counts": true,
  "run_us": true,
  "run_pr": false,
  "reader": "csv",
  "state_filter_us": ["01", "04"],
  "privacy_defn": "puredp"
}
"""

US_THRESHOLD_KEYS = [
    "(USA, 1)",
    "(USA, 2)",
    "(STATE, 1)",
    "(STATE, 2)",
    "(COUNTY, 1)",
    "(COUNTY, 2)",
    "(TRACT, 1)",
    "(TRACT, 2)",
    "(PLACE, 1)",
    "(PLACE, 2)",
    "(AIANNH, 1)",
    "(AIANNH, 2)",
]

PR_THRESHOLD_KEYS = [
    "(PR-STATE, 1)",
    "(PR-STATE, 2)",
    "(PR-COUNTY, 1)",
    "(PR-COUNTY, 2)",
    "(PR-TRACT, 1)",
    "(PR-TRACT, 2)",
    "(PR-PLACE, 1)",
    "(PR-PLACE, 2)",
]

# pylint: enable=line-too-long

ALL_THRESHOLD_KEYS = US_THRESHOLD_KEYS + PR_THRESHOLD_KEYS

NEGATIVE_THRESHOLDS = {key: [-1000000, 2000000, 3000000] for key in ALL_THRESHOLD_KEYS}

INFINITE_THRESHOLDS = {
    key: [float("inf"), float("inf"), float("inf")] for key in ALL_THRESHOLD_KEYS
}


def write_input_files(
    data_dir: str,
    persons: str,
    ethnicity_iterations: str,
    race_iterations: str,
    race_eth_codes: str,
    iteration_map: str,
    grfc: str,
    config: str,
):
    """Writes a complete set of input files to a directory."""
    with open(os.path.join(data_dir, "GRF-C.txt"), "w") as grfc_file:
        grfc_file.write(grfc)

    with open(os.path.join(data_dir, "person-records.txt"), "w") as perons_file:
        perons_file.write(persons)

    config_dir = os.path.join(data_dir, "config")
    os.mkdir(config_dir)

    with open(os.path.join(config_dir, "config.json"), "w") as config_file:
        config_file.write(config)

    with open(
        os.path.join(config_dir, "ethnicity-characteristic-iterations.txt"), "w"
    ) as ethnicity_file:
        ethnicity_file.write(ethnicity_iterations)

    with open(
        os.path.join(config_dir, "race-characteristic-iterations.txt"), "w"
    ) as race_file:
        race_file.write(race_iterations)

    with open(
        os.path.join(config_dir, "race-and-ethnicity-codes.txt"), "w"
    ) as race_eth_codes_file:
        race_eth_codes_file.write(race_eth_codes)

    with open(
        os.path.join(config_dir, "race-and-ethnicity-code-to-iteration.txt"), "w"
    ) as iteration_map_file:
        iteration_map_file.write(iteration_map)


class TestThresholding(PySparkTest):
    """Tests for thresholding, measured on the output files."""

    @parameterized.expand([param("with noise", True), param("no noise", False)])
    @attr("slow")
    def test_thresholding_drops_zeroes_t1(self, _, noise: bool):
        """Tests that thresholding drops zero values with high probability.

        This test is non-deterministic with a small failure probability <1e-10."""

        # Create empty input file.
        data_dir = tempfile.TemporaryDirectory()
        write_input_files(
            data_dir=data_dir.name,
            persons=STATE_FILTERABLE_INPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_AND_ETHNICITY_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_US_GRFC,
            config=SIMPLE_CONFIG,
        )

        output_dir = tempfile.TemporaryDirectory()

        if noise:
            budget = TEN_PRIVACY_BUDGET_P
        else:
            budget = INFINITE_PRIVACY_BUDGET_P

        run_plan_p_analytics(
            os.path.join(data_dir.name, "config"),
            data_dir.name,
            output_dir.name,
            overwrite_config={
                "zero_suppression_chance": 0.99999,
                "thresholds_p": INFINITE_THRESHOLDS,
                **budget,
            },
        )

        t1_df = multi_read_csv(os.path.join(output_dir.name, "t1"), dtype=str, sep="|")

        # Removing State = 04 (and regions within) as it has a non zero population.
        t1_df_filtered = t1_df[t1_df["REGION_ID"].isin(US_CHECKED_REGIONS)]

        t1_national_state_df = t1_df_filtered[
            t1_df_filtered.REGION_TYPE.isin(["USA", "STATE"])
        ]
        t1_substate_df = t1_df_filtered[
            ~t1_df_filtered.REGION_TYPE.isin(["USA", "STATE"])
        ]

        # Thresholding should only be applied below the state level.
        t1_national_state_expected = pd.DataFrame(
            {
                "REGION_ID": ["1", "1", "1", "1", "01", "01", "01", "01"],
                "REGION_TYPE": [
                    "USA",
                    "USA",
                    "USA",
                    "USA",
                    "STATE",
                    "STATE",
                    "STATE",
                    "STATE",
                ],
                "ITERATION_CODE": [
                    "3010",
                    "3011",
                    "2001",
                    "3001",
                    "3010",
                    "3011",
                    "2001",
                    "3001",
                ],
            }
        )

        self.assert_frame_equal_with_sort(
            t1_national_state_df[["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]],
            t1_national_state_expected,
        )

        # This test is randomized. All rows are zeroes, so each row will be dropped
        # with probability >= 0.99999. We've set the thresholds so it's unlikely that
        # any rows will get age/sex breakdowns. This means we expect that before
        # thresholding there will be:
        #
        # 3 common iteration codes * 4 geographies +
        # 1 detailed iteration code * 0 geographies = 12 rows.
        #
        # The odds of passing the threshold more than 3 times is < 1e-10,
        # which we judge an acceptable failure probability, while still showing that
        # most of the rows are thresholded away.
        # If there is no noise, then we should always get 0 rows.
        if noise:
            maximum_rows = 3
        else:
            maximum_rows = 0
        self.assertLessEqual(len(t1_substate_df.index), maximum_rows)

    @parameterized.expand([param("with noise", True), param("no noise", False)])
    @attr("slow")
    def test_thresholding_drops_pr_zeroes_t1(self, _, noise: bool):
        """Tests that thresholding drops zero values in PR with high probability.

        This test is non-deterministic with a small failure probability <1e-10.
        """

        # Create empty input file.
        data_dir = tempfile.TemporaryDirectory()
        write_input_files(
            data_dir=data_dir.name,
            persons=STATE_FILTERABLE_INPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_AND_ETHNICITY_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_PR_GRFC,
            config=SIMPLE_CONFIG,
        )

        output_dir = tempfile.TemporaryDirectory()

        if noise:
            budget = TEN_PRIVACY_BUDGET_P
        else:
            budget = INFINITE_PRIVACY_BUDGET_P

        run_plan_p_analytics(
            os.path.join(data_dir.name, "config"),
            data_dir.name,
            output_dir.name,
            overwrite_config={
                "zero_suppression_chance": 0.99999,
                "thresholds_p": INFINITE_THRESHOLDS,
                **budget,
                "run_us": False,
                "run_pr": True,
            },
        )

        t1_df = multi_read_csv(os.path.join(output_dir.name, "t1"), dtype=str, sep="|")

        # Removing County = 002 (and regions within) as it has a non zero population
        t1_df_filtered = t1_df[t1_df["REGION_ID"].isin(PR_CHECKED_REGIONS)]

        t1_state_df = t1_df_filtered[t1_df_filtered.REGION_TYPE.isin(["PR-STATE"])]
        t1_substate_df = t1_df_filtered[~t1_df_filtered.REGION_TYPE.isin(["PR-STATE"])]

        # Thresholding should only be applied below the state level.
        t1_state_expected = pd.DataFrame(
            {
                "REGION_ID": ["72", "72", "72", "72"],
                "REGION_TYPE": ["PR-STATE", "PR-STATE", "PR-STATE", "PR-STATE"],
                "ITERATION_CODE": ["3010", "3011", "2001", "3001"],
            }
        )
        self.assert_frame_equal_with_sort(
            t1_state_df[["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]],
            t1_state_expected,
        )

        # This test is randomized. All rows are zeroes, so each row will be dropped
        # with probability >= 0.99999. We've set the thresholds so it's unlikely that
        # any rows will get age/sex breakdowns. This means we expect that before
        # thresholding there will be:
        #
        # 3 common iteration codes * 4 geographies +
        # 1 detailed iteration code * 0 geographies = 12 rows.
        #
        # The odds of passing the threshold more than 3 times is < 1e-10,
        # which we judge an acceptable failure probability, while still showing that
        # most of the rows are thresholded away.
        # If there is no noise, then we should always get 0 rows.
        if noise:
            maximum_rows = 3
        else:
            maximum_rows = 0
        self.assertLessEqual(len(t1_substate_df.index), maximum_rows)

    @parameterized.expand(
        [
            param("with noise", TEN_PRIVACY_BUDGET_P),
            param("no noise", INFINITE_PRIVACY_BUDGET_P),
        ]
    )
    @attr("slow")
    def test_thresholding_ignores_t2(self, _, budget):
        """Tests that thresholding does not drop small values from the t2 counts.

        This noisy version of thistest is non-deterministic with a small failure
        probability.
        """

        # Create empty input file.
        data_dir = tempfile.TemporaryDirectory()
        write_input_files(
            data_dir=data_dir.name,
            persons=STATE_FILTERABLE_INPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_AND_ETHNICITY_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_US_GRFC,
            config=SIMPLE_CONFIG,
        )

        output_dir = tempfile.TemporaryDirectory()

        run_plan_p_analytics(
            os.path.join(data_dir.name, "config"),
            data_dir.name,
            output_dir.name,
            overwrite_config={
                "zero_suppression_chance": 0.99999,
                "thresholds_p": NEGATIVE_THRESHOLDS,
                **budget,
            },
        )

        t2_df = multi_read_csv(
            os.path.join(output_dir.name, "t2"),
            dtype=str,
            sep="|",
            usecols=["REGION_ID", "COUNT"],
        )

        # Removing State = 04 (and regions within) as it has a non zero population.
        t2_df_filtered = t2_df[t2_df["REGION_ID"].isin(US_CHECKED_REGIONS)]

        # The noise version of this test is randomized, but we expect every cell
        # to pass the threshold for a t2 count (age4) with very high probability.
        # In the version with no noise, we know every cell will pass the threshold.
        # Given that, we expect:
        #
        # 3 common iteration codes * 6 geographies * (4 age buckets + 1 total row) *
        # 2 sex codes = 180 rows.
        # No threshold is applied to these rows, so we expect all to be present.
        self.assertEqual(len(t2_df_filtered.index), 180)

    @parameterized.expand(
        [
            param("with noise", TEN_PRIVACY_BUDGET_P),
            param("no noise", INFINITE_PRIVACY_BUDGET_P),
        ]
    )
    @attr("slow")
    def test_no_thresholding_unspecified(self, _, budget):
        """Tests nothing is dropped when zero_suppression_chance isn't specified."""
        # Create empty input file.
        data_dir = tempfile.TemporaryDirectory()
        write_input_files(
            data_dir=data_dir.name,
            persons=STATE_FILTERABLE_INPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_AND_ETHNICITY_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_US_GRFC,
            config=CONFIG_NO_SUPPRESSION,
        )

        output_dir = tempfile.TemporaryDirectory()

        run_plan_p_analytics(
            os.path.join(data_dir.name, "config"),
            data_dir.name,
            output_dir.name,
            overwrite_config=budget,
        )

        t1_df = multi_read_csv(
            os.path.join(output_dir.name, "t1"),
            dtype=str,
            sep="|",
            usecols=["REGION_ID", "COUNT"],
        )

        # Removing State = 04 (and regions within) as it has a non zero population.
        t1_df_filtered = t1_df[t1_df["REGION_ID"].isin(US_CHECKED_REGIONS)]

        # There is no thresholding, so we expect every t1 cell to be included.
        # In the version with no noise, we know every cell will pass the threshold.
        # We expect:
        #
        # 3 common iteration codes * 6 geographies + 1 detail_only iterations * 2 high
        # level geographies = 20 rows.
        self.assertEqual(len(t1_df_filtered.index), 20)

    @parameterized.expand(
        [
            param("with noise", TEN_PRIVACY_BUDGET_P),
            param("no noise", INFINITE_PRIVACY_BUDGET_P),
        ]
    )
    @attr("slow")
    def test_no_thresholding(self, _, budget):
        """Tests that zero_suppression_chance = 0.0 disables thresholding."""
        # Create empty input file.
        data_dir = tempfile.TemporaryDirectory()
        write_input_files(
            data_dir=data_dir.name,
            persons=STATE_FILTERABLE_INPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_AND_ETHNICITY_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_US_GRFC,
            config=SIMPLE_CONFIG,
        )

        output_dir = tempfile.TemporaryDirectory()

        run_plan_p_analytics(
            os.path.join(data_dir.name, "config"),
            data_dir.name,
            output_dir.name,
            overwrite_config={**budget, "zero_suppression_chance": 0.0},
        )

        t1_df = multi_read_csv(
            os.path.join(output_dir.name, "t1"),
            dtype={"REGION_ID": "str", "COUNT": "int"},
            sep="|",
            usecols=["REGION_ID", "COUNT"],
        )

        # Removing State = 04 (and regions within) as it has a non zero population
        t1_df_filtered = t1_df[t1_df["REGION_ID"].isin(US_CHECKED_REGIONS)]

        # There is no thresholding, so we expect every t1 cell to be included.
        # In the version with no noise, we know every cell will pass the threshold.
        # We expect:
        #
        # 3 common iteration codes * 6 geographies + 1 detail_only iterations * 2 high
        # level geographies = 20 rows.
        self.assertEqual(len(t1_df_filtered.index), 20)


@parameterized_class(
    [
        {"parameters_input_dir": "input_dir_puredp"},
        {"parameters_input_dir": "input_dir_zcdp"},
    ]
)
class TestAlgorithmAlone(PySparkTest):
    """Test algorithm output by itself."""

    parameters_input_dir: str

    def setUp(self):
        """Create temporary directories."""

        self.output_files = ["t1", "t2"]
        self.data_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.data_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.output_dir = tempfile.TemporaryDirectory()

    @parameterized.expand([(False,), (True,)])
    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime.
    def test_non_negativity_post_processing(self, allow_negative_counts_flag: bool):
        """SafeTab-P eliminates negative values in output based on flag.

        Args:
            allow_negative_counts_flag: Whether or not negative counts are allowed.
        """
        run_plan_p_analytics(
            os.path.join(self.data_path, self.parameters_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={
                "thresholds_p": MANUAL_THRESHOLDS,
                "allow_negative_counts": allow_negative_counts_flag,
                # Turn off thresholding because it will also drop negative counts.
                "zero_suppression_chance": 0.0,
            },
        )
        for output_file in self.output_files:
            print(f"Checking {output_file}")
            df = multi_read_csv(
                os.path.join(self.output_dir.name, output_file),
                dtype=int,
                sep="|",
                usecols=["COUNT"],
            )
            contains_negatives = any(df["COUNT"] < 0)
            self.assertEqual(contains_negatives, allow_negative_counts_flag)

    @attr("slow")  # This test is not run frequently as it takes longer than 10 minutes.
    def test_output_format(self):
        """SafeTab-P output files pass validation.

        See resources/config/output, and `Appendix A` for details about the expected
        output formats.
        """
        run_plan_p_analytics(
            os.path.join(self.data_path, self.parameters_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={
                "run_us": True,
                "run_pr": True,
                "allow_negative_counts": True,
            },
            should_validate_private_output=True,
        )
        for output_file in self.output_files:
            self.assertTrue(
                validate_directory_single_config(
                    os.path.join(self.output_dir.name, output_file),
                    os.path.join(RESOURCES_DIR, f"config/output/{output_file}.json"),
                    delimiter="|",
                )
            )

    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime.
    def test_exclude_states(self):
        """SafeTab-P can exclude specific states from tabulation."""
        include_states = ["02"]
        run_plan_p_analytics(
            os.path.join(self.data_path, self.parameters_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={
                "thresholds_p": MANUAL_THRESHOLDS,
                "state_filter_us": include_states,
                # Turn off thresholding because it can interfere with state exclusion.
                "zero_suppression_chance": 0.0,
            },
        )

        for output_file in self.output_files:
            print(f"Checking {output_file}")
            df = multi_read_csv(
                os.path.join(self.output_dir.name, output_file),
                dtype=str,
                sep="|",
                usecols=["REGION_TYPE", "REGION_ID"],
            )
            df = df[df["REGION_TYPE"] == "STATE"]
            actual = df["REGION_ID"].unique()
            self.assertEqual(actual, include_states)


@parameterized_class(
    [
        {"parameters_input_dir": "input_dir_puredp"},
        {"parameters_input_dir": "input_dir_zcdp"},
    ]
)
class TestAlgorithmsUSandPR(PySparkTest):
    """Test algorithm runs on US and/or Puerto Rico."""

    parameters_input_dir: str

    def setUp(self):
        """Create temporary directories."""

        self.output_files = ["t1", "t2"]
        self.data_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.data_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.output_dir = tempfile.TemporaryDirectory()

    @parameterized.expand([(False, True), (True, False), (True, True)])
    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_run_us_run_pr(self, run_us: bool, run_pr: bool):
        """SafeTab-P can run for US, PR or both.

        Args:
            run_us: Whether to run SafeTab-P on the US geographies.
            run_pr: Whether to run SafeTab-P on Puerto Rico.
        """
        run_plan_p_analytics(
            os.path.join(self.data_path, self.parameters_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={
                "thresholds_p": MANUAL_THRESHOLDS,
                "run_us": run_us,
                "run_pr": run_pr,
                # PR only has one record in the toy dataset, so it's likely to be
                # thresholded.
                "zero_suppression_chance": 0.0,
            },
        )
        for output_file in self.output_files:
            print(f"Checking {output_file}")
            df = multi_read_csv(
                os.path.join(self.output_dir.name, output_file),
                dtype=str,
                sep="|",
                usecols=["REGION_TYPE", "REGION_ID"],
            )
            self.assertEqual("STATE" in df["REGION_TYPE"].values, run_us)
            self.assertEqual("PR-STATE" in df["REGION_TYPE"].values, run_pr)

    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime
    def test_run_us_run_pr_both_false(self):
        """SafeTab-P fails if run_us and run_pr are False."""
        with self.assertRaisesRegex(
            ValueError,
            "Invalid config: At least one of 'run_us', 'run_pr' must be True.",
        ):
            run_plan_p_analytics(
                os.path.join(self.data_path, self.parameters_input_dir),
                self.data_path,
                self.output_dir.name,
                overwrite_config={"run_us": False, "run_pr": False},
            )


@parameterized_class(
    [
        {"name": "safetab_p_puredp", "parameters_dir_name": "input_dir_puredp"},
        {"name": "safetab_p_zcdp", "parameters_dir_name": "input_dir_zcdp"},
    ]
)
class TestPopulationGroups(PySparkTest):
    """Test that SafeTab-P tabulates the correct population groups."""

    parameters_dir_name: str

    def setUp(self):
        self.output_files = ["t1", "t2"]
        self.reader_config_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.reader_config_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.parameters_path = os.path.join(self.data_path, self.parameters_dir_name)
        self.output_dir = tempfile.TemporaryDirectory()
        self.output_path = self.output_dir.name

    @parameterized.expand(
        [
            (True, False, ["01", "02", "11"]),
            (False, True, ["01", "02", "11"]),
            (True, True, ["01", "02", "11"]),
            (True, True, ["01"]),
        ]
    )
    @attr("slow")
    def test_pop_groups(self, run_us, run_pr, include_states):
        """Test that SafeTab-P tabulates the correct population groups."""

        # Read iterations.
        race_iterations_df = pd.read_csv(
            os.path.join(self.parameters_path, "race-characteristic-iterations.txt"),
            delimiter="|",
            dtype=str,
        ).drop("ALONE", axis="columns")
        ethnicity_iterations_df = pd.read_csv(
            os.path.join(
                self.parameters_path, "ethnicity-characteristic-iterations.txt"
            ),
            delimiter="|",
            dtype=str,
        )
        iterations_df = pd.concat([race_iterations_df, ethnicity_iterations_df])
        iterations_df = iterations_df[iterations_df["LEVEL"] != "0"]

        # Read geographies.
        grfc_df = pd.read_csv(
            os.path.join(self.data_path, "GRF-C.txt"), delimiter="|", dtype=str
        )

        pr_states = ["72"] if run_pr else []
        us_states = include_states if run_us else []
        us_grfc_df = grfc_df[grfc_df["TABBLKST"].isin(us_states)]
        pr_grfc_df = grfc_df[grfc_df["TABBLKST"].isin(pr_states)]
        usa_geos = [("USA", "1")] if run_us else []
        state_geos = [("STATE", x) for x in us_states] + [
            ("PR-STATE", x) for x in pr_states
        ]
        county_geos = [
            ("COUNTY", x) for x in set(us_grfc_df["TABBLKST"] + us_grfc_df["TABBLKCOU"])
        ] + [
            ("PR-COUNTY", x)
            for x in set(pr_grfc_df["TABBLKST"] + pr_grfc_df["TABBLKCOU"])
        ]
        tract_geos = [
            ("TRACT", x)
            for x in set(
                us_grfc_df["TABBLKST"]
                + us_grfc_df["TABBLKCOU"]
                + us_grfc_df["TABTRACTCE"]
            )
        ] + [
            ("PR-TRACT", x)
            for x in set(
                pr_grfc_df["TABBLKST"]
                + pr_grfc_df["TABBLKCOU"]
                + pr_grfc_df["TABTRACTCE"]
            )
        ]
        place_geos = [
            ("PLACE", x)
            for x in filter(
                lambda x: x[2:] != "99999",
                set(us_grfc_df["TABBLKST"] + us_grfc_df["PLACEFP"]),
            )
        ] + [
            ("PR-PLACE", x)
            for x in filter(
                lambda x: x[2:] != "99999",
                set(pr_grfc_df["TABBLKST"] + pr_grfc_df["PLACEFP"]),
            )
        ]
        aiannh_geos = (
            [("AIANNH", x) for x in set(us_grfc_df["AIANNHCE"]) - {"9999"}]
            if run_us
            else []
        )
        geo_df = pd.DataFrame(
            usa_geos + state_geos + county_geos + tract_geos + place_geos + aiannh_geos,
            columns=["REGION_TYPE", "REGION_ID"],
        )

        pop_groups_df = iterations_df.merge(geo_df, how="cross")
        pop_groups_df = pop_groups_df[
            ~(
                (pop_groups_df["DETAILED_ONLY"] == "True")
                & pop_groups_df["REGION_TYPE"].isin(
                    [
                        "PR-COUNTY",
                        "PR-TRACT",
                        "PR-PLACE",
                        "COUNTY",
                        "TRACT",
                        "PLACE",
                        "AIANNH",
                    ]
                )
            )
        ]

        if "t1" in self.output_files:
            pop_groups_df = pop_groups_df[
                ~(
                    (pop_groups_df["COARSE_ONLY"] == "True")
                    & pop_groups_df["REGION_TYPE"].isin(["USA", "PR-STATE", "STATE"])
                )
            ]
            pop_groups_df = pop_groups_df[
                ~(
                    (pop_groups_df["LEVEL"] == "1")
                    & (
                        pop_groups_df["REGION_TYPE"].isin(
                            ["PR-TRACT", "TRACT", "PR-PLACE", "PLACE", "AIANNH"]
                        )
                    )
                )
            ]

        run_plan_p_analytics(
            self.parameters_path,
            self.data_path,
            self.output_path,
            overwrite_config={
                "run_us": run_us,
                "run_pr": run_pr,
                "state_filter_us": include_states,
                # Thresholding will remove groups that should be included.
                "zero_suppression_chance": 0.0,
            },
        )

        actual_dfs = []
        for output_file in self.output_files:
            actual_dfs.append(
                multi_read_csv(
                    os.path.join(self.output_path, output_file),
                    dtype=str,
                    sep="|",
                    usecols=["REGION_TYPE", "REGION_ID", "ITERATION_CODE"],
                )
            )
        actual_df = pd.concat(actual_dfs).drop_duplicates()

        self.assert_frame_equal_with_sort(
            actual_df,
            pop_groups_df.loc[:, ["REGION_TYPE", "REGION_ID", "ITERATION_CODE"]],
        )


@parameterized_class(
    [
        {"name": "safetab_p_puredp", "parameters_dir_name": "input_dir_puredp"},
        {"name": "safetab_p_zcdp", "parameters_dir_name": "input_dir_zcdp"},
    ]
)
class TestStatisticsGranularity(PySparkTest):
    """Test that SafeTab-P tabulates the correct statistics granularity."""

    parameters_dir_name: str

    def setUp(self):
        self.reader_config_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.reader_config_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.parameters_path = os.path.join(self.data_path, self.parameters_dir_name)
        self.actual_dir = tempfile.TemporaryDirectory()
        self.actual_path = self.actual_dir.name
        self.expected_dir = tempfile.TemporaryDirectory()
        self.expected_path = self.expected_dir.name

        self.config_dir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.parameters_path, "config.json"), "r") as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            self.us_or_puerto_rico = "US"  # This test runs US only.
            state_filter = config_json[STATE_FILTER_FLAG]
            validate_input(  # Create configs used to run SafeTab.
                parameters_path=self.parameters_path,
                input_data_configs_path=INPUT_CONFIG_DIR,
                output_path=self.config_dir.name,
                program="safetab-p",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=self.data_path,
                    state_filter=state_filter,
                    program="safetab-p",
                ),
                state_filter=state_filter,
            )

    @parameterized.expand([(list(range(3)),), ([2, 4, 6],)])
    @attr("slow")
    def test_statistics_granularity(self, thresholds):
        """Test that SafeTab-P tabulates the correct statistics granularity."""
        # Set manual thresholds.
        manual_thresholds = {key: thresholds for key in US_THRESHOLD_KEYS}

        # Read iterations.
        race_iterations_df = pd.read_csv(
            os.path.join(self.parameters_path, "race-characteristic-iterations.txt"),
            delimiter="|",
            dtype=str,
        ).drop("ALONE", axis="columns")
        ethnicity_iterations_df = pd.read_csv(
            os.path.join(
                self.parameters_path, "ethnicity-characteristic-iterations.txt"
            ),
            delimiter="|",
            dtype=str,
        )
        iterations_df = pd.concat([race_iterations_df, ethnicity_iterations_df])
        iterations_df = iterations_df[iterations_df["LEVEL"] != "0"]

        # Run ground truth algorithm and actual algorithm with budget=inf.
        create_ground_truth_p(
            parameters_path=self.parameters_path,
            data_path=self.data_path,
            output_path=self.expected_path,
            config_path=self.config_dir.name,
            overwrite_config=INFINITE_PRIVACY_BUDGET_P,
            us_or_puerto_rico=self.us_or_puerto_rico,
        )
        execute_plan_p_analytics(
            self.parameters_path,
            self.data_path,
            self.actual_path,
            config_path=self.config_dir.name,
            overwrite_config={
                **INFINITE_PRIVACY_BUDGET_P,
                "thresholds_p": manual_thresholds,
                "zero_suppression_chance": 0.0,
            },
            us_or_puerto_rico=self.us_or_puerto_rico,
        )

        # Get expected stat levels from ground truth t1.
        ground_truth = multi_read_csv(
            os.path.join(self.expected_path, "t1"), dtype=str, sep="|"
        )
        ground_truth["COUNT"] = ground_truth["COUNT"].astype(int)
        ground_truth["STAT_LEVEL"] = np.digitize(ground_truth["COUNT"], thresholds)
        ground_truth = ground_truth.merge(
            iterations_df.loc[:, ["ITERATION_CODE", "DETAILED_ONLY"]]
        )
        ground_truth.loc[ground_truth.DETAILED_ONLY == "True", "STAT_LEVEL"] = 0
        ground_truth_stat_levels = ground_truth.drop(["COUNT", "DETAILED_ONLY"], axis=1)

        # Get actual t2 stat levels.
        actual_t2 = multi_read_csv(
            os.path.join(self.actual_path, "t2"), dtype=str, sep="|"
        )

        actual_t2_stat_levels = (
            actual_t2.groupby(["REGION_TYPE", "REGION_ID", "ITERATION_CODE"])
            .size()
            .map(WORKLOAD_SIZE_TO_STAT_LEVEL)
            .to_frame(name="STAT_LEVEL")
            .reset_index()
        )
        actual_t2_stat_levels["STAT_LEVEL"] = actual_t2_stat_levels[
            "STAT_LEVEL"
        ].astype(int)

        # Get actual t1 stat levels.
        actual_t1 = multi_read_csv(
            os.path.join(self.actual_path, "t1"), dtype=str, sep="|"
        )
        actual_t1_stat_levels = actual_t1.drop(["COUNT"], axis=1)

        # Figure out the t2 sex marginal rows and drop these from noisy t1 as we
        # already acoount for it in noisy t2.
        t2_sex_marginal_rows = actual_t2.query("AGESTART == '*' & AGEEND == '*'")
        t2_sex_marginal_rows = t2_sex_marginal_rows[
            ["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]
        ]
        actual_t1_stat_levels = (
            actual_t1_stat_levels.merge(
                t2_sex_marginal_rows,
                on=["REGION_ID", "REGION_TYPE", "ITERATION_CODE"],
                how="left",
                indicator=True,
            )
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )
        actual_t1_stat_levels["STAT_LEVEL"] = 0

        # Combine t1/t2 stat levels.
        actual_stat_levels = actual_t1_stat_levels.append(
            actual_t2_stat_levels, ignore_index=True
        )

        # Compare stat levels.
        self.assert_frame_equal_with_sort(actual_stat_levels, ground_truth_stat_levels)


@parameterized_class([{"parameters_input_dir": "input_dir_puredp"}])
class TestInvalidInput(PySparkTest):
    """Verify that SafeTab-P raises exceptions on invalid input."""

    parameters_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        self.data_path = str(os.path.join(RESOURCES_DIR, "toy_dataset"))
        self.config_dir = tempfile.TemporaryDirectory()
        with open(
            os.path.join(
                os.path.join(self.data_path, self.parameters_input_dir), "config.json"
            ),
            "r",
        ) as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            # This test just runs US because execute_plan_p_analytics defaults to US.
            _ = validate_state_filter_us(config_json[STATE_FILTER_FLAG])
            state_filter = config_json[STATE_FILTER_FLAG]
            validate_input(  # Create configs used to run SafeTab.
                parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
                input_data_configs_path=INPUT_CONFIG_DIR,
                output_path=self.config_dir.name,
                program="safetab-p",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=self.data_path,
                    state_filter=state_filter,
                    program="safetab-p",
                ),
                state_filter=state_filter,
            )

    @parameterized.expand(
        [
            ("s3://dummy", "dummy/"),
            ("dummy/", "s3://dummy"),
            ("s3://dummy", "s3://dummy"),
            ("s3a://dummy", "dummy/"),
            ("dummy/", "s3a://dummy"),
            ("s3a://dummy", "s3a://dummy"),
        ]
    )
    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime.
    def test_read_from_s3_spark_local(self, input_path: str, output_path: str):
        """SafeTab-P fails if input directory is on s3 and in Spark local mode.

        Args:
            input_path: The directory that contains the CEF files and parameters
            directory.
            output_path: The path where the output should be saved.
        """
        with self.assertRaisesRegex(
            RuntimeError,
            "Reading and writing to and from s3"
            " is not supported when running Spark in local mode.",
        ):
            execute_plan_p_analytics(
                parameters_path=input_path,
                data_path=input_path,
                output_path=output_path,
                config_path="dummy_config_dir",
            )

    @parameterized.expand([(-1.0,), (0.0,), (1.0,), (2.0,)])
    @attr("slow")
    # This test is not run frequently based on the criticality of the test and runtime.
    def test_invalid_stage_1_fraction(self, test_fraction: float):
        """SafeTab-P fails if 'privacy_budget_p_stage_1_fraction is not between 0 and 1.

        Args:
            test_fraction: The invalid stage 1 fraction to test.
        """
        with self.assertRaisesRegex(
            ValueError,
            "Invalid config: 'privacy_budget_p_stage_1_fraction' must be between 0"
            " and 1.",
        ):
            execute_plan_p_analytics(
                parameters_path=os.path.join(self.data_path, self.parameters_input_dir),
                data_path=self.data_path,
                output_path="",
                config_path=self.config_dir.name,
                overwrite_config={
                    "privacy_budget_p_stage_1_fraction": test_fraction,
                    "thresholds_p": MANUAL_THRESHOLDS,
                },
            )
