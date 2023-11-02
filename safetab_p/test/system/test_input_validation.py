"""Tests SafeTab-P input validation."""

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

import json
import os
import re
import shutil
import tempfile
from typing import List
from unittest.mock import Mock, patch

import pandas as pd
from parameterized import parameterized
from pyspark.sql import SparkSession

from tmlt.analytics._schema import Schema, analytics_to_spark_schema
from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.paths import RESOURCES_DIR
from tmlt.safetab_utils.input_schemas import GEO_SCHEMA, PERSON_SCHEMA
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)

# pylint: disable=no-member


class TestInputValidation(PySparkTest):
    """Tests for SafeTab-P input validation."""

    def setUp(self):
        """Set up test."""
        self.program = "safetab-p"
        self.records_filename = "person-records.txt"
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        self.parameters_path = os.path.join(self.data_path, "input_dir_puredp")
        self.output_dir = tempfile.TemporaryDirectory()
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.config_path = os.path.join(RESOURCES_DIR, "config/input")
        # Get params to create input_reader
        with open(os.path.join(self.parameters_path, "config.json"), "r") as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            # Using default of execute_plan_p which is US.
            _ = validate_state_filter_us(config_json[STATE_FILTER_FLAG])
            self.state_filter = config_json[STATE_FILTER_FLAG]
        self.input_reader = safetab_input_reader(
            reader, self.data_path, self.state_filter, "safetab-p"
        )

    def test_valid_input(self):
        """Toy input is parsed successfully."""
        with self.assertLogs(level="INFO") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertTrue(okay)
        self.assertIn(
            "INFO:tmlt.safetab_utils.input_validation:"
            "Phase 3 successful. All files are as expected.",
            logs.output,
        )

    @patch("tmlt.safetab_utils.utils.CSVPReader", autospec=True)
    def test_invalid_input_filter_states(self, mock_reader):
        """Input is bad and states are not filtered.

        Tests that invalid input returned by CSV reader fails validation.
        """
        spark = SparkSession.builder.getOrCreate()
        invalid_person_sdf = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    (
                        42,
                        "1",
                        "True",
                        "01",
                        "001",
                        "000001",
                        "0002",
                        "03",
                        "1011",
                        "1012",
                        "1031",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "1010",
                    ),
                    (
                        73,
                        "1",
                        "True",
                        "11",
                        "001",
                        "000100",
                        "3004",
                        "01",
                        "3310",
                        "5454",
                        "2322",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "7680",
                    ),
                    (
                        79,
                        "1",
                        "True",
                        "04",
                        "031",
                        "050901",
                        "2040",
                        "01",
                        "1340",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "Null",
                        "5230",
                    ),
                ]
            ),
            analytics_to_spark_schema(Schema(PERSON_SCHEMA)),
        )
        invalid_geo_sdf = spark.createDataFrame(
            spark.sparkContext.parallelize(
                [
                    ("01", "001", "000001", "0002", "22800", "0001"),
                    ("11", "001", "000100", "3004", "99999", "9999"),
                    ("04", "031", "050901", "2040", "99999", "9999"),
                ]
            ),
            analytics_to_spark_schema(Schema(GEO_SCHEMA)),
        )
        state_filter = ["01", "11"]
        mock_reader.get_person_df = Mock(return_value=invalid_person_sdf)
        mock_reader.get_geo_df = Mock(return_value=invalid_geo_sdf)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                mock_reader,
                self.parameters_path,
                self.config_path,
                self.output_dir.name,
                self.program,
                state_filter,
            )
            self.assertFalse(okay)
            self.assertIn(
                "ERROR:root:Invalid values found in TABBLKST: ['04']", logs.output
            )
            self.assertIn(
                "ERROR:root:person-records failed schema validation.", logs.output
            )
            self.assertIn("ERROR:root:GRF-C failed schema validation.", logs.output)
            self.assertIn(
                "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 3. See"
                " above.",
                logs.output,
            )

    def test_invalid_format_phase1_failure(self):
        """Input validation fails if the format is invalid."""
        filename = os.path.join(self.data_path, self.records_filename)
        df = pd.read_csv(filename, sep="|", dtype=str)
        qrace1_loc = df.columns.get_loc("QRACE1")
        df.iloc[1, qrace1_loc] = "Does not conform to format"
        df.to_csv(filename, sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    def test_outside_domain_phase3_failure(self):
        """Input validation fails if some values are outside the domain."""
        filename = os.path.join(self.data_path, self.records_filename)
        df = pd.read_csv(filename, sep="|", dtype=str)
        qrace1_loc = df.columns.get_loc("QRACE1")
        df.iloc[1, qrace1_loc] = "9999"
        df.to_csv(filename, sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 3. See"
            " above.",
            logs.output,
        )

    def test_utf8_compatible(self):
        """Input validation fails if the input is not uft8 compatible."""
        filename = os.path.join(self.parameters_path, "race-and-ethnicity-codes.txt")
        df = pd.read_csv(filename, sep="|")
        race_name_loc = df.columns.get_loc("RACE_ETH_NAME")
        df.iloc[0, race_name_loc] = "¥"
        df.to_csv(filename, encoding="iso-8859-1", sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        expected_output = re.escape(
            "ERROR:root:'utf-8' codec can't decode byte 0xa5 in position"
        )
        self.assertTrue(
            any(
                re.match(expected_output, actual_output)
                for actual_output in logs.output
            )
        )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    @parameterized.expand(
        [
            (
                "race-characteristic-iterations.txt",
                pd.DataFrame(
                    [["1230", "Name1", "1", "False", "False", "True"]],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [["1240", "Name2", "1", "False", "True", "True"]],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
            (
                "ethnicity-characteristic-iterations.txt",
                pd.DataFrame(
                    [["1230", "Name1", "1", "False", "True"]],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [["1240", "Name2", "1", "True", "True"]],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
        ]
    )
    def test_detailed_only_and_coarse_only(
        self, filename: str, okay_df: pd.DataFrame, bad_df: pd.DataFrame
    ):
        """Fails if any iterations are DETAILED_ONLY and COARSE_ONLY.

        Args:
            filename: The name of the file where the race/ethnicity codes will be
                written.
            okay_df: A DataFrame containing valid records.
            bad_df: A DataFrame containing invalid records.
        """
        df = pd.concat([okay_df, bad_df], ignore_index=True)
        filename = os.path.join(self.parameters_path, filename)
        df.to_csv(filename, sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:"
            f"{len(bad_df)} of {len(df)} iterations had DETAILED_ONLY and COARSE_ONLY"
            f" as 'True' in {filename}",
            logs.output,
        )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    @parameterized.expand(
        [
            (
                "race-characteristic-iterations.txt",
                pd.DataFrame(
                    [
                        ["1230", "Name1", "0", "False", "Null", "Null"],
                        ["1240", "Name2", "1", "False", "False", "True"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [
                        ["1250", "Name3", "1", "False", "Null", "Null"],
                        ["1260", "Name4", "1", "False", "Null", "True"],
                        ["1270", "Name5", "1", "False", "False", "Null"],
                        ["1280", "Name6", "0", "False", "Null", "True"],
                        ["1290", "Name7", "0", "False", "False", "Null"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
            (
                "ethnicity-characteristic-iterations.txt",
                pd.DataFrame(
                    [
                        ["1230", "Name1", "0", "Null", "Null"],
                        ["1240", "Name2", "1", "False", "True"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [
                        ["1250", "Name3", "1", "Null", "Null"],
                        ["1260", "Name4", "1", "Null", "True"],
                        ["1270", "Name5", "1", "False", "Null"],
                        ["1280", "Name6", "0", "Null", "True"],
                        ["1290", "Name7", "0", "False", "Null"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
        ]
    )
    def test_iterations_with_inconsistent_Null_and_level(
        self, filename: str, okay_df: pd.DataFrame, bad_df: pd.DataFrame
    ):
        """Iterations have Null for detailed/coarse only if LEVEL=='0'.

        Args:
            filename: The name of the file where the race/ethnicity codes will be
                written.
            okay_df: A DataFrame containing valid records.
            bad_df: A DataFrame containing invalid records.
        """
        df = pd.concat([okay_df, bad_df], ignore_index=True)
        filename = os.path.join(self.parameters_path, filename)
        df.to_csv(filename, sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            f"ERROR:tmlt.safetab_utils.input_validation:{len(bad_df)} of {len(df)}"
            " iterations did not have matching LEVEL == '0', DETAILED_ONLY == 'Null'"
            f", and COARSE_ONLY == 'Null' in {filename}",
            logs.output,
        )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    def test_error_race_code_after_null(self):
        """Input validation errors if there are race codes after a 'Null'."""
        filename = os.path.join(self.data_path, self.records_filename)
        df = pd.read_csv(filename, sep="|", dtype=str)
        race_codes = ["1010", "Null", "1012"] + ["Null"] * 5
        columns = [f"QRACE{i + 1}" for i in range(8)]
        for race_code, column in zip(race_codes, columns):
            column_loc = df.columns.get_loc(column)
            df.iloc[1, column_loc] = race_code
            df.iloc[4, column_loc] = race_code
        df.to_csv(filename, sep="|", index=False)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:2 records had race codes "
            "after a 'Null'.",
            logs.output,
        )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    @parameterized.expand(
        [
            (
                "race-characteristic-iterations.txt",
                pd.DataFrame(
                    [
                        [
                            "1140",
                            "Albanian alone or in any combination",
                            "2",
                            "False",
                            "False",
                            "False",
                        ],
                        ["2001", "European alone", "1", "True", "False", "False"],
                        [
                            "9999",
                            "Dummy Albanian Iteration",
                            "2",
                            "False",
                            "False",
                            "False",
                        ],
                        ["9998", "Dummy2", "1", "True", "True", "False"],
                        ["9997", "Dummy3", "1", "True", "True", "False"],
                        ["9996", "Dummy4", "1", "True", "True", "False"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [
                        ["1140", "1010"],
                        ["2001", "1020"],
                        ["2001", "1031"],
                        ["9999", "1010"],
                        ["9998", "1020"],
                        ["9997", "1020"],
                        ["9996", "1031"],
                    ],
                    columns=["ITERATION_CODE", "RACE_ETH_CODE"],
                ),
                [
                    "The following race codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=2 and ALONE=False, and"
                    " are all tabulated at the national and state level. SafeTab"
                    " expects each code to only be mapped to one such iteration"
                    " code.\n\nThe list of errors is formatted as [race code]:"
                    " [iteration codes].\n1010: 1140, 9999\n",
                    "The following race codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=2 and ALONE=False, and"
                    " are all tabulated at the sub-state level. SafeTab expects"
                    " each code to only be mapped to one such iteration"
                    " code.\n\nThe list of errors is formatted as [race code]:"
                    " [iteration codes].\n1010: 1140, 9999\n",
                    "The following race codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=1 and ALONE=True, and are"
                    " all tabulated at the national and state level. SafeTab"
                    " expects each code to only be mapped to one such iteration"
                    " code.\n\nThe list of errors is formatted as [race code]:"
                    " [iteration codes].\n1020: 2001, 9997, 9998\n1031: 2001, 9996\n",
                ],
            ),
            (
                "ethnicity-characteristic-iterations.txt",
                pd.DataFrame(
                    [
                        ["3010", "Central American", "1", "False", "False"],
                        ["3011", "Costa Rican", "2", "False", "False"],
                        ["9999", "Dummy Costa Rican Iteration", "2", "False", "False"],
                        ["9998", "Dummy2", "1", "True", "False"],
                        ["9997", "Dummy2", "1", "True", "False"],
                        ["9996", "Dummy2", "1", "True", "False"],
                        ["9995", "Dummy2", "1", "False", "False"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [
                        ["3010", "2010"],
                        ["3010", "2020"],
                        ["3011", "2020"],
                        ["9999", "2020"],
                        ["9998", "2010"],
                        ["9997", "2010"],
                        ["9996", "2011"],
                        ["9995", "2011"],
                    ],
                    columns=["ITERATION_CODE", "RACE_ETH_CODE"],
                ),
                [
                    "The following ethnicity codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=2, and are all tabulated"
                    " at the national and state level. SafeTab expects each code to"
                    " only be mapped to one such iteration code.\n\nThe list of"
                    " errors is formatted as [ethnicity code]: [iteration"
                    " codes].\n2020: 3011, 9999\n",
                    "The following ethnicity codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=2, and are all tabulated"
                    " at the sub-state level. SafeTab expects each code to only be"
                    " mapped to one such iteration code.\n\nThe list of errors is"
                    " formatted as [ethnicity code]: [iteration codes].\n2020:"
                    " 3011, 9999\n",
                    "The following ethnicity codes are mapped to multiple iteration"
                    " codes that are all labeled as LEVEL=1, and are all tabulated"
                    " at the national and state level. SafeTab expects each code to"
                    " only be mapped to one such iteration code.\n\nThe list of"
                    " errors is formatted as [ethnicity code]: [iteration"
                    " codes].\n2010: 3010, 9997, 9998\n2011: 9995, 9996\n",
                ],
            ),
        ]
    )
    def test_check_iteration_hierarchy_height(
        self,
        filename: str,
        iterations: pd.DataFrame,
        mappings: pd.DataFrame,
        error_messages: List[str],
    ):
        """Validation errors when inferred levels is too many.

        Args:
            filename: The name of the file containing existing race/ethnicity
                iterations.
            iterations: A dataframe containing the iterations.
            mappings: A dataframe containing the race/ethnicity code to iteration
                mappings.
            error_messages: A list of expected error messages.
        """
        iterations.to_csv(
            os.path.join(self.parameters_path, filename), sep="|", index=False
        )
        mappings.to_csv(
            os.path.join(
                self.parameters_path, "race-and-ethnicity-code-to-iteration.txt"
            ),
            sep="|",
            index=False,
        )

        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )

        self.assertFalse(okay)
        for error_message in error_messages:
            self.assertIn(
                "ERROR:tmlt.safetab_utils.input_validation:" + error_message,
                logs.output,
            )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    [
                        [
                            "9999",
                            "Dummy Albanian Iteration",
                            "2",
                            "False",
                            "False",
                            "False",
                        ]
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [["9999", "Dummy Costa Rican Iteration", "2", "False", "False"]],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
            (
                pd.DataFrame(
                    [
                        [
                            "9999",
                            "Dummy Albanian Iteration",
                            "2",
                            "False",
                            "False",
                            "False",
                        ],
                        [
                            "9998",
                            "Dummy Race Iteration",
                            "2",
                            "False",
                            "False",
                            "False",
                        ],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "ALONE",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
                pd.DataFrame(
                    [
                        ["9999", "Dummy Costa Rican Iteration", "2", "False", "False"],
                        ["9998", "Dummy Hispanic Iteration", "2", "False", "False"],
                    ],
                    columns=[
                        "ITERATION_CODE",
                        "ITERATION_NAME",
                        "LEVEL",
                        "DETAILED_ONLY",
                        "COARSE_ONLY",
                    ],
                ),
            ),
        ]
    )
    def test_check_race_and_ethnicity_iteration_codes_are_disjoint(
        self,
        extra_race_iterations: pd.DataFrame,
        extra_ethnicity_iterations: pd.DataFrame,
    ):
        """Validation errors when race and ethnicity iterations are not disjoint.

        Args:
          extra_race_iterations: New race iterations to add.
          extra_ethnicity_iterations: New ethnicity iterations to add.
        """
        pd.concat(
            [
                pd.read_csv(
                    os.path.join(
                        self.parameters_path, "race-characteristic-iterations.txt"
                    ),
                    sep="|",
                    dtype=str,
                ),
                extra_race_iterations,
            ]
        ).to_csv(
            os.path.join(self.parameters_path, "race-characteristic-iterations.txt"),
            sep="|",
            index=False,
        )
        pd.concat(
            [
                pd.read_csv(
                    os.path.join(
                        self.parameters_path, "ethnicity-characteristic-iterations.txt"
                    ),
                    sep="|",
                    dtype=str,
                ),
                extra_ethnicity_iterations,
            ]
        ).to_csv(
            os.path.join(
                self.parameters_path, "ethnicity-characteristic-iterations.txt"
            ),
            sep="|",
            index=False,
        )

        intersect_list = sorted(
            list(
                set(extra_race_iterations["ITERATION_CODE"]).intersection(
                    set(extra_ethnicity_iterations["ITERATION_CODE"])
                )
            )
        )

        with self.assertLogs(level="ERROR") as logs:
            okay = validate_input(
                input_reader=self.input_reader,
                parameters_path=self.parameters_path,
                input_data_configs_path=self.config_path,
                output_path=self.output_dir.name,
                program=self.program,
                state_filter=self.state_filter,
            )
        self.assertFalse(okay)
        self.assertIn(
            f"ERROR:tmlt.safetab_utils.input_validation:{intersect_list} appeared as "
            "race iteration codes and as ethnicity iteration codes."
            " Expected race and ethnicity iteration codes to be disjoint.",
            logs.output,
        )
        self.assertIn(
            "ERROR:tmlt.safetab_utils.input_validation:Errors found in phase 1. See"
            " above.",
            logs.output,
        )
