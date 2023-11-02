"""Tests SafeTab-P output validation."""

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

from io import StringIO
from typing import List

import pandas as pd
from parameterized import parameterized
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.paths import (
    get_safetab_p_output_configs,
    setup_safetab_p_output_config_dir,
)
from tmlt.safetab_utils.output_validation import validate_output

# pylint: disable=no-member


class TestOutputValidation(PySparkTest):
    """Tests for SafeTab-P output validation."""

    def setUp(self):
        """Set up test."""
        self.sp = SparkSession.builder.getOrCreate()
        t2_schema = StructType(
            [
                StructField(c, StringType())
                for c in [
                    "REGION_ID",
                    "REGION_TYPE",
                    "ITERATION_CODE",
                    "AGESTART",
                    "AGEEND",
                    "SEX",
                    "COUNT",
                ]
            ]
        )
        data = """REGION_ID|REGION_TYPE|ITERATION_CODE|AGESTART|AGEEND|SEX|COUNT
                          1|        USA|          3674|       0|     4|  1|    2
                          1|        USA|          3674|       0|     4|  2|    3
        """
        data = data.replace(" ", "")
        self.t2_sdf = self.sp.createDataFrame(
            pd.read_csv(StringIO(data), sep="|"), schema=t2_schema
        )
        setup_safetab_p_output_config_dir()

    @parameterized.expand(
        [
            (
                ["72"],
                True,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", 5],
                        ["1", "USA", "2472", 3],
                        ["720000000", "PR-STATE", "3977", 8],  # Invalid REGION_ID.
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
                "Not as per expected format. See above.",
            ),
            (
                ["72"],
                True,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", -5],
                        ["1", "USA", "2472", -3],
                        ["72", "PR-STATE", "23977", -8],  # Invalid ITERATION_CODE.
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
                "Not as per expected format. See above.",
            ),
            (
                ["72"],
                True,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", -5],
                        ["1", "USA", "2472", 3],
                        ["11000", "PR-COUNTY", "3977", 8],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
                "Output does not have state filters applied.",
            ),
            (
                ["72"],
                False,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", -5],
                        ["1", "USA", "2472", 3],
                        ["72", "PR-STATE", "3977", 8],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
                "Negative counts are not allowed, but were found in the output.",
            ),
        ]
    )
    def test_invalid_output(
        self,
        state_filter: List[str],
        allow_negative_counts_flag: bool,
        t1_df: pd.DataFrame,
        error_message: str,
    ):
        """Output validation fails on invalid output and logs appropriate error."""
        t1_sdf = self.sp.createDataFrame(t1_df)
        with self.assertLogs(level="ERROR") as logs:
            okay = validate_output(
                output_sdfs={"t1": t1_sdf, "t2": self.t2_sdf},
                expected_output_configs=get_safetab_p_output_configs(),
                state_filter=state_filter,
                allow_negative_counts_flag=allow_negative_counts_flag,
            )
        self.assertFalse(okay)
        self.assertIn(
            "ERROR:tmlt.safetab_utils.output_validation:Invalid output:"
            f" {error_message}",
            logs.output,
        )

    @parameterized.expand(
        [
            (
                ["72"],
                True,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", 5],
                        ["1", "USA", "2472", 3],
                        ["72", "PR-STATE", "3977", 8],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
            ),
            (
                ["72"],
                True,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", -5],
                        ["1", "USA", "2472", -3],
                        ["72", "PR-STATE", "3977", 8],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
            ),
            (
                ["72"],
                False,
                pd.DataFrame(
                    [
                        ["1", "USA", "3674", 5],
                        ["1", "USA", "2472", 3],
                        ["72", "PR-STATE", "3977", 8],
                    ],
                    columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
                ),
            ),
        ]
    )
    def test_valid_output(
        self,
        state_filter: List[str],
        allow_negative_counts_flag: bool,
        t1_df: pd.DataFrame,
    ):
        """Output validation passes for valid outputs."""
        t1_sdf = self.sp.createDataFrame(t1_df)
        with self.assertLogs(level="INFO") as logs:
            okay = validate_output(
                output_sdfs={"t1": t1_sdf, "t2": self.t2_sdf},
                expected_output_configs=get_safetab_p_output_configs(),
                state_filter=state_filter,
                allow_negative_counts_flag=allow_negative_counts_flag,
            )
        self.assertTrue(okay)
        self.assertIn(
            "INFO:tmlt.safetab_utils.output_validation:Output validation successful."
            " All output files are as expected.",
            logs.output,
        )
