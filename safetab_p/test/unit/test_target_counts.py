"""Unit tests for :mod:`tmlt.safetab_p.target_counts`."""

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
import tempfile

import pandas as pd

from tmlt.common.io_helpers import to_csv_with_create_dir
from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.target_counts_p import get_stat_level_from_safetab_output


class TestTargetCounts(PySparkTest):
    """TestCases for target counts."""

    def test_get_stat_level_from_safetab_output(self):  # pylint: disable=no-self-use
        """:func:`get_stat_level_from_safetab_output` is correct on test input."""
        safetab_dir1 = tempfile.TemporaryDirectory()
        safetab_dir2 = tempfile.TemporaryDirectory()

        to_csv_with_create_dir(
            pd.DataFrame(
                [["01", "A", "1001", 0], ["02", "B", "2001", 0]],
                columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
            ),
            os.path.join(safetab_dir1.name, "t1", "t1.csv"),
            sep="|",
            index=False,
        )
        to_csv_with_create_dir(
            pd.DataFrame(
                columns=[
                    "REGION_ID",
                    "REGION_TYPE",
                    "ITERATION_CODE",
                    "AGESTART",
                    "AGEEND",
                    "SEX",
                    "COUNT",
                ]
            ),
            os.path.join(safetab_dir1.name, "t2", "t2.csv"),
            sep="|",
            index=False,
        )

        to_csv_with_create_dir(
            pd.DataFrame(
                [["02", "B", "2001", 0]],
                columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
            ),
            os.path.join(safetab_dir2.name, "t1", "t1.csv"),
            sep="|",
            index=False,
        )
        to_csv_with_create_dir(
            pd.DataFrame(
                [
                    ["01", "A", "1001", "0", "17", "1", 0],
                    ["01", "A", "1001", "18", "44", "1", 0],
                    ["01", "A", "1001", "45", "64", "1", 0],
                    ["01", "A", "1001", "65", "115", "1", 0],
                    ["01", "A", "1001", "*", "*", "1", 0],
                    ["01", "A", "1001", "0", "17", "2", 0],
                    ["01", "A", "1001", "18", "44", "2", 0],
                    ["01", "A", "1001", "45", "64", "2", 0],
                    ["01", "A", "1001", "65", "115", "2", 0],
                    ["01", "A", "1001", "*", "*", "2", 0],
                ],
                columns=[
                    "REGION_ID",
                    "REGION_TYPE",
                    "ITERATION_CODE",
                    "AGESTART",
                    "AGEEND",
                    "SEX",
                    "COUNT",
                ],
            ),
            os.path.join(safetab_dir2.name, "t2", "t2.csv"),
            sep="|",
            index=False,
        )

        expected_df = pd.DataFrame(
            [["01", "A", "1001", "1"], ["02", "B", "2001", "0"]],
            columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "STAT_LEVEL"],
        )
        actual_df = get_stat_level_from_safetab_output(
            [safetab_dir1.name, safetab_dir2.name]
        ).toPandas()
        self.assert_frame_equal_with_sort(expected_df, actual_df)
