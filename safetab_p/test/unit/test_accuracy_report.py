"""Unit tests for :mod:`tmlt.safetab_p.accuracy_report`."""

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

# pylint: disable=no-self-use

import os
import tempfile

import pandas as pd

from tmlt.common.io_helpers import to_csv_with_create_dir
from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.accuracy_report import (
    create_aggregated_error_report_p,
    create_error_report_p,
)


class TestSingleRunAccuracyReports(PySparkTest):
    """TestCase for single run accuracy reports for SafeTab-P."""

    def setUp(self):
        """Create shared input files."""
        self.parameters_dir = tempfile.TemporaryDirectory()
        self.noisy_dir = tempfile.TemporaryDirectory()
        self.target_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()

        pd.DataFrame(
            [["1001", "False", "False"]],
            columns=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
        ).to_csv(
            os.path.join(
                self.parameters_dir.name, "race-characteristic-iterations.txt"
            ),
            sep="|",
            index=False,
        )

        pd.DataFrame(
            [["2001", "True", "False"]],
            columns=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
        ).to_csv(
            os.path.join(
                self.parameters_dir.name, "ethnicity-characteristic-iterations.txt"
            ),
            sep="|",
            index=False,
        )

    def test_create_error_report_p(self):
        """create_error_report_p creates the expected output."""
        t1_df = pd.DataFrame(
            [["01", "A", "2001", 2], ["01", "A", "1001", 1]],
            columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
        )
        to_csv_with_create_dir(
            t1_df,
            os.path.join(self.target_dir.name, "t1", "t1.csv"),
            sep="|",
            index=False,
        )

        t2_df = pd.DataFrame(
            [
                ["02", "B", "1001", "0", "17", "1", 1],
                ["02", "B", "1001", "18", "44", "1", 1],
                ["02", "B", "1001", "45", "64", "1", 1],
                ["02", "B", "1001", "65", "115", "1", 1],
                ["02", "B", "1001", "0", "17", "2", 0],
                ["02", "B", "1001", "18", "44", "2", 0],
                ["02", "B", "1001", "45", "64", "2", 0],
                ["02", "B", "1001", "65", "115", "2", 0],
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
        )
        to_csv_with_create_dir(
            t2_df,
            os.path.join(self.target_dir.name, "t2", "t2.csv"),
            sep="|",
            index=False,
        )

        t1_df = pd.DataFrame(
            [["01", "A", "2001", 2], ["01", "A", "1001", 3]],
            columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"],
        )
        to_csv_with_create_dir(
            t1_df,
            os.path.join(self.noisy_dir.name, "t1", "t1.csv"),
            sep="|",
            index=False,
        )

        t2_df = pd.DataFrame(
            [
                ["02", "B", "1001", "0", "17", "1", -1],
                ["02", "B", "1001", "18", "44", "1", 4],
                ["02", "B", "1001", "45", "64", "1", 1],
                ["02", "B", "1001", "65", "115", "1", 2],
                ["02", "B", "1001", "0", "17", "2", 0],
                ["02", "B", "1001", "18", "44", "2", 0],
                ["02", "B", "1001", "45", "64", "2", 0],
                ["02", "B", "1001", "65", "115", "2", 0],
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
        )
        to_csv_with_create_dir(
            t2_df,
            os.path.join(self.noisy_dir.name, "t2", "t2.csv"),
            sep="|",
            index=False,
        )

        expected_df = pd.DataFrame(
            [
                ["01", "A", "1001", "False", "False", "1", "Total", "2", "4"],
                ["01", "A", "2001", "True", "False", "2", "Total", "0", "0"],
                ["02", "B", "1001", "False", "False", "4", "Sex x Age(4)", "6", "14"],
            ],
            columns=[
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DETAILED_ONLY",
                "COARSE_ONLY",
                "Population group size",
                "Workload selected",
                "Total absolute error",
                "Total squared error",
            ],
        )

        create_error_report_p(
            noisy_path=self.noisy_dir.name,
            ground_truth_path=self.target_dir.name,
            parameters_path=self.parameters_dir.name,
            output_path=self.output_dir.name,
        )

        actual_df = pd.read_csv(
            os.path.join(self.output_dir.name, "error_report.csv"), dtype=str
        )

        pd.testing.assert_frame_equal(
            expected_df.set_index(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]),
            actual_df.set_index(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]),
        )


class TestMultiRunSafeTabPAccuracyReport(PySparkTest):
    """TestCase for SafeTab-P multi-run error report."""

    def setUp(self):
        """Set up input."""
        self.parameters_dir = tempfile.TemporaryDirectory()
        self.noisy_run_dir_1 = tempfile.TemporaryDirectory()
        self.noisy_run_dir_2 = tempfile.TemporaryDirectory()
        self.ground_truth_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()

        # Set up iterations files.
        with open(
            os.path.join(
                self.parameters_dir.name, "race-characteristic-iterations.txt"
            ),
            "w",
        ) as f:
            f.write("ITERATION_CODE|DETAILED_ONLY|COARSE_ONLY|LEVEL\n")
            f.write("1001|False|False|1\n")
            f.write("2001|False|False|1\n")
        with open(
            os.path.join(
                self.parameters_dir.name, "ethnicity-characteristic-iterations.txt"
            ),
            "w",
        ) as f:
            f.write("ITERATION_CODE|DETAILED_ONLY|COARSE_ONLY|LEVEL\n")
            f.write("3001|True|False|2\n")

        # Set up ground truth.
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t1"))
        with open(os.path.join(self.ground_truth_dir.name, "t1", "t1.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT\n")
            f.write("01|A|1001|4\n")
            f.write("02|A|2001|1\n")
            f.write("03|B|3001|0\n")
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t2"))
        with open(os.path.join(self.ground_truth_dir.name, "t2", "t2.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|AGESTART|AGEEND|SEX|COUNT\n")
            f.write("01|A|1001|0|17|1|1\n")
            f.write("01|A|1001|18|44|1|1\n")
            f.write("01|A|1001|45|64|1|1\n")
            f.write("01|A|1001|65|115|1|1\n")
            f.write("01|A|1001|0|17|2|0\n")
            f.write("01|A|1001|18|44|2|0\n")
            f.write("01|A|1001|45|64|2|0\n")
            f.write("01|A|1001|65|115|2|0\n")

        # Set up noisy counts.
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t1"))
        with open(os.path.join(self.noisy_run_dir_1.name, "t1", "t1.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT\n")
            f.write("02|A|2001|4\n")
            f.write("03|B|3001|1\n")
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t2"))
        with open(os.path.join(self.noisy_run_dir_1.name, "t2", "t2.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|AGESTART|AGEEND|SEX|COUNT\n")
            f.write("01|A|1001|0|17|1|2\n")
            f.write("01|A|1001|18|44|1|3\n")
            f.write("01|A|1001|45|64|1|4\n")
            f.write("01|A|1001|65|115|1|5\n")
            f.write("01|A|1001|0|17|2|6\n")
            f.write("01|A|1001|18|44|2|-7\n")
            f.write("01|A|1001|45|64|2|8\n")
            f.write("01|A|1001|65|115|2|9\n")
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t1"))
        with open(os.path.join(self.noisy_run_dir_2.name, "t1", "t1.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT\n")
            f.write("01|A|1001|-1\n")
            f.write("02|A|2001|5\n")
            f.write("03|B|3001|2\n")
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t2"))
        with open(os.path.join(self.noisy_run_dir_2.name, "t2", "t2.csv"), "w") as f:
            f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|AGESTART|AGEEND|SEX|COUNT\n")

    def test_aggregate_error_report(self):
        """create_aggregate_error_report_p creates the expected output."""
        expected_df = pd.DataFrame(
            [
                [
                    "A",
                    "1",
                    "False",
                    "[0.0, 1.0]",
                    "1.00",
                    "3.95",
                    "0.00",
                    None,
                    "0.00",
                    None,
                    "0.00",
                    None,
                ],
                [
                    "A",
                    "1",
                    "False",
                    "(1.0, 10.0]",
                    "0.50",
                    "5.00",
                    "0.50",
                    "8.65",
                    "0.00",
                    None,
                    "0.00",
                    None,
                ],
                [
                    "B",
                    "2",
                    "True",
                    "[0.0, 1.0]",
                    "1.00",
                    "1.95",
                    "0.00",
                    None,
                    "0.00",
                    None,
                    "0.00",
                    None,
                ],
            ],
            columns=[
                "REGION_TYPE",
                "ITERATION_LEVEL",
                "DETAILED_ONLY",
                "Population group size",
                "Average proportion of pop groups with workload = Total",
                "MOE of Total",
                "Average proportion of pop groups with workload = Sex x Age(4)",
                "MOE of Sex x Age(4)",
                "Average proportion of pop groups with workload = Sex x Age(9)",
                "MOE of Sex x Age(9)",
                "Average proportion of pop groups with workload = Sex x Age(23)",
                "MOE of Sex x Age(23)",
            ],
        )

        create_aggregated_error_report_p(
            single_run_paths=[self.noisy_run_dir_1.name, self.noisy_run_dir_2.name],
            parameters_path=self.parameters_dir.name,
            ground_truth_path=self.ground_truth_dir.name,
            output_path=self.output_dir.name,
        )
        actual_df = pd.read_csv(
            os.path.join(self.output_dir.name, "multi_run_error_report.csv"), dtype=str
        )
        pd.testing.assert_frame_equal(
            expected_df.set_index(
                [
                    "REGION_TYPE",
                    "ITERATION_LEVEL",
                    "DETAILED_ONLY",
                    "Population group size",
                ]
            ),
            actual_df.set_index(
                [
                    "REGION_TYPE",
                    "ITERATION_LEVEL",
                    "DETAILED_ONLY",
                    "Population group size",
                ]
            ),
        )
