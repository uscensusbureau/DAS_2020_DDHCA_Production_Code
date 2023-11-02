"""Unit tests for :mod:`tmlt.common.error_report`."""

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

# pylint: disable=protected-access

import unittest
from functools import partial
from unittest.mock import Mock, call

import numpy as np
import pandas as pd

from tmlt.common.error_report import (
    ERROR_COLUMN,
    NAME_COLUMN,
    AggregatedErrorMetric,
    ErrorReport,
    suppress_warnings,
)


class TestAggregatedErrorMetric(unittest.TestCase):
    """Unit tests for :class:`tmlt.common.error_report.AggregatedErrorMetric`."""

    def setUp(self):
        """Set up test."""

        self.metric = AggregatedErrorMetric(
            name="average_absolute_error",
            error_metric=lambda nondp, dp: abs(nondp["estimate"] - dp["estimate"]),
            aggregation=partial(np.mean, axis=1),
            filters=[("query_type", "count"), ("measure_column", "")],
        )
        self.combined_df = pd.DataFrame(
            {
                "A": ["a1", "a1", "a2", "a2"],
                "B": ["b1", "b2", "b1", "b2"],
                "query_type": ["count"] * 4,
                "measure_column": [""] * 4,
                "estimate": [1, 2, 3, 0.3],
                "estimate_0": [0.7, 2.5, 2.7, 0.6],
                "estimate_1": [1.3, 1.5, 3.3, -0.3],
                "CI_lower": [1, 2, 3, 0.3],
                "CI_lower_0": [0.6, 2.1, 1.9, 0.5],
                "CI_lower_1": [0, 0, 0, 0],
                "CI_upper": [1, 2, 3, 0.3],
                "CI_upper_0": [0.8, 2.9, 3.5, 0.7],
                "CI_upper_1": [10, 10, 10, 10],
            }
        )
        self.expected = pd.DataFrame(
            [
                ["a1", "b1", "count", "", "average_absolute_error", 0.3],
                ["a1", "b2", "count", "", "average_absolute_error", 0.5],
                ["a2", "b1", "count", "", "average_absolute_error", 0.3],
                ["a2", "b2", "count", "", "average_absolute_error", 0.45],
            ],
            columns=[
                "A",
                "B",
                "query_type",
                "measure_column",
                NAME_COLUMN,
                ERROR_COLUMN,
            ],
        )
        self.query_columns = ["A", "B", "query_type", "measure_column"]
        self.result_columns = ["estimate", "CI_lower", "CI_upper"]
        self.trials = 2

    def test_aggregated_error_metric(self):
        """AggregatedErrorMetric returns the expected result."""
        expected = self.expected
        actual = self.metric(
            self.combined_df, self.query_columns, self.result_columns, self.trials
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_ignore_extra_columns(self):
        """AggregatedErrorMetrics ignores columns it is not told about."""
        combined_df = self.combined_df
        combined_df["C"] = "c1"
        expected = self.expected
        actual = self.metric(
            self.combined_df, self.query_columns, self.result_columns, self.trials
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_filter_removes_rows(self):
        """AggregatedErrorMetrics ignores rows excluded by filter."""
        combined_df = self.combined_df
        combined_df.append(
            {
                "A": "a1",
                "B": "a2",
                "query_type": "median",
                "measure_column": "C",
                "estimate": 0,
                "estimate_0": 0,
                "estimate_1": 0,
            },
            ignore_index=True,
        )
        expected = self.expected
        actual = self.metric(
            self.combined_df, self.query_columns, self.result_columns, self.trials
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_multiple_columns(self):
        """AggregatedErrorMetrics can use multiple columns in the error metrics."""

        def within_CI(nondp: pd.DataFrame, dp: pd.DataFrame) -> pd.Series:
            """Return whether the true answer was within the CI for each row.

            Args:
                nondp: The confidence intervals for the true, non-DP answers, with
                    columns 'CI_lower' and 'CI_upper' specifying the intervals.
                dp: DP estimates of the answers, with column 'estimate' containing
                    the estimates.
            """
            return (dp["CI_lower"] <= nondp["estimate"]) & (
                nondp["estimate"] <= dp["CI_upper"]
            )

        metric = AggregatedErrorMetric(
            name="proportion_within_CI",
            error_metric=within_CI,
            aggregation=partial(np.mean, axis=1),
        )
        actual = metric(
            self.combined_df, self.query_columns, self.result_columns, self.trials
        )
        expected = pd.DataFrame(
            [
                ["a1", "b1", "count", "", "proportion_within_CI", 0.5],
                ["a1", "b2", "count", "", "proportion_within_CI", 0.5],
                ["a2", "b1", "count", "", "proportion_within_CI", 1.0],
                ["a2", "b2", "count", "", "proportion_within_CI", 0.5],
            ],
            columns=[
                "A",
                "B",
                "query_type",
                "measure_column",
                NAME_COLUMN,
                ERROR_COLUMN,
            ],
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestErrorReport(unittest.TestCase):
    """Tests for :class:`tmlt.common.error_report.ErrorReport`."""

    def setUp(self):
        """Set up test."""
        columns = [
            "A",
            "B",
            "query_type",
            "measure_column",
            "estimate",
            "estimate_0",
            "estimate_1",
        ]
        self.combined_df = pd.DataFrame(
            [
                ["a1", "b1", "count", "", 0, 0.7, 1.3],
                ["a1", "b1", "Q1", "X", np.nan, 1.5, 2.2],
                ["a1", "b2", "count", "", 1, 1.4, -2.1],
                ["a1", "b2", "Q1", "X", 2.5, np.nan, 4],
                ["a2", "b1", "count", "", 25, 22, 24],
                ["a2", "b1", "Q1", "X", 3.5, 3, 4],
                ["a2", "b2", "count", "", 4, 3.6, 4.3],
                ["a2", "b2", "Q1", "X", 2, 3.1, 4.1],
            ],
            columns=columns,
        )

        self.runner = Mock()
        self.runner.query_columns = ["A", "B", "query_type", "measure_column"]
        self.runner.result_columns = ["estimate"]
        self.runner.run_nondp.return_value = self.combined_df[
            ["A", "B", "query_type", "measure_column", "estimate"]
        ].copy()
        self.runner.run_dp.side_effect = [
            self.combined_df[
                ["A", "B", "query_type", "measure_column", "estimate_0"]
            ].copy(),
            self.combined_df[
                ["A", "B", "query_type", "measure_column", "estimate_1"]
            ].copy(),
        ]
        self.error_report = ErrorReport(self.runner)
        self.eps = 1
        self.trials = 2

    def test_create_combined_df(self):
        """Error report combines multiple trials correctly."""
        actual = self.error_report._create_combined_df(self.eps, self.trials)
        expected = self.combined_df
        pd.testing.assert_frame_equal(actual, expected)

    def test_error_report(self):
        """Error report returns the expected output."""
        metrics = [
            AggregatedErrorMetric(
                name="median_absolute_error",
                error_metric=lambda a, b: abs(a["estimate"] - b["estimate"]),
                aggregation=suppress_warnings(partial(np.nanmedian, axis=1)),
                filters=[("query_type", "Q1"), ("measure_column", "X")],
            ),
            AggregatedErrorMetric(
                name="dp_nan_count",
                error_metric=lambda a, b: b["estimate"].isna(),
                aggregation=partial(np.sum, axis=1),
                filters=[("query_type", "Q1"), ("measure_column", "X")],
            ),
            AggregatedErrorMetric(
                name="average_absolute_error",
                error_metric=lambda a, b: abs(a["estimate"] - b["estimate"]),
                aggregation=partial(np.mean, axis=1),
                filters=[("query_type", "count"), ("measure_column", "")],
            ),
        ]
        self.error_report._create_combined_df = Mock()
        self.error_report._create_combined_df.return_value = self.combined_df
        expected = pd.DataFrame(
            [
                ["a1", "b1", "Q1", "X", "median_absolute_error", np.nan],
                ["a1", "b2", "Q1", "X", "median_absolute_error", 1.5],
                ["a2", "b1", "Q1", "X", "median_absolute_error", 0.5],
                ["a2", "b2", "Q1", "X", "median_absolute_error", 1.6],
                ["a1", "b1", "Q1", "X", "dp_nan_count", 0.0],
                ["a1", "b2", "Q1", "X", "dp_nan_count", 1.0],
                ["a2", "b1", "Q1", "X", "dp_nan_count", 0.0],
                ["a2", "b2", "Q1", "X", "dp_nan_count", 0.0],
                ["a1", "b1", "count", "", "average_absolute_error", 1.0],
                ["a1", "b2", "count", "", "average_absolute_error", 1.75],
                ["a2", "b1", "count", "", "average_absolute_error", 2.0],
                ["a2", "b2", "count", "", "average_absolute_error", 0.35],
            ],
            columns=["A", "B", "query_type", "measure_column", "error_metric", "error"],
        )
        actual = self.error_report(metrics, self.eps, self.trials)
        pd.testing.assert_frame_equal(actual, expected)

    def test_error_report_seed(self):
        """Error report runs mechanism with incrementally increasing seeds."""
        metrics = [
            AggregatedErrorMetric(
                name="median_absolute_error",
                error_metric=lambda a, b: abs(a["estimate"] - b["estimate"]),
                aggregation=suppress_warnings(partial(np.nanmedian, axis=1)),
                filters=[("query_type", "Q1"), ("measure_column", "X")],
            )
        ]
        self.error_report(metrics, self.eps, self.trials)
        self.runner.run_dp.assert_has_calls(
            [call(self.eps, seed=0), call(self.eps, seed=1)]
        )
