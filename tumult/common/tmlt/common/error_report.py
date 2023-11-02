"""Tool for creating error reports.

Combines multiple trials of a DP algorithm and compares the results against
a run using a non-DP version of the algorithm.
"""

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

import warnings
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NAME_COLUMN = "error_metric"
"""Name of column to store name of each error metric."""

ERROR_COLUMN = "error"
"""Name of column to store aggregated errors."""


class Runner(ABC):
    """Standardized interface for running algorithms with tabular output.

    Notice that by default run_nondp is simply run_dp with infinite eps, but
    can be overwritten with an alternate implementation.
    """

    @property
    @abstractmethod
    def query_columns(self) -> List[str]:
        """Return a list of columns describing the query each row represents."""

    @property
    @abstractmethod
    def result_columns(self) -> List[str]:
        """Return a list of columns describing the query output in each row."""

    @abstractmethod
    def run_dp(self, eps: float, seed: Optional[int] = None) -> pd.DataFrame:
        """Run the DP algorithm and return the results as a DataFrame.

        Grouping columns are type str,
        Statistics columns are type float (may be Nan).

        Args:
            eps: The epsilon budget.
            seed: A seed for the mechanism.
        """

    def run_nondp(self) -> pd.DataFrame:
        """Run the non-DP algorithm and return the results as a DataFrame.

        Grouping columns are type str,
        Statistics columns are type float (may be Nan).
        """
        return self.run_dp(np.inf)


class AggregatedErrorMetric:
    """Calculates arbitrary error metrics aggregated over multiple trials."""

    def __init__(
        self,
        name: str,
        error_metric: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
        aggregation: Callable[[np.ndarray], np.ndarray],
        filters: Iterable[Tuple[str, str]] = None,
    ):
        """Constructor.

        Args:
            name: A name for the error metric.
            error_metric: A callable that takes in a pd.DataFrame of non-dp
                result_columns (first argument) and a pd.DataFrame of dp result_columns
                (second argument) and returns the error as a pd.Series with the same
                length as the input dataframe. Common examples include squared error
                and relative error.
            aggregation: A callable that aggregates errors over trials. Takes in a 2d
                np.ndarray array (1 row for each cell, 1 column for each trial) and
                returns the aggregated error for each cell. Common examples include
                mean and standard deviation.
            filters: A list of column_name, column_value pairs, determining what subset
                of the output to compute the error on.
        """
        self._name = name
        self._filters = filters if filters else []
        self._error_metric = error_metric
        self._aggregation = aggregation

    def __call__(
        self,
        combined_df: pd.DataFrame,
        query_columns: List[str],
        result_columns: Sequence[str],
        trials: int,
    ) -> pd.DataFrame:
        """Return a DataFrame containing the aggregated errors.

        Args:
            combined_df: A dataframe with columns for query_columns, as
                well as {result_column}_0, {result_column}_1, etc for each trial, for
                each result_column.
            query_columns: A list of columns describing the query each row represents.
            result_columns: A list of columns describing the query output in each row.
            trials: The number of trials to run.

        Returns:
            A DataFrame containing the aggregated errors. The following columns are
            included

            * query_columns
            * :data:`NAME_COLUMN`
            * :data:`ERROR_COLUMN`
        """
        for column, value in self._filters:
            combined_df = combined_df[combined_df[column] == value]
        if combined_df.empty:
            return pd.DataFrame(columns=query_columns + [NAME_COLUMN, ERROR_COLUMN])
        errors = np.zeros((len(combined_df), trials), dtype=float)

        # Calculate errors for every row in each run
        nondp_df = combined_df[result_columns]
        for i in range(trials):
            dp_trial_result_columns = [
                f"{result_column}_{i}" for result_column in result_columns
            ]
            dp_trial_df = combined_df[dp_trial_result_columns].rename(
                columns=dict(zip(dp_trial_result_columns, result_columns))
            )
            errors[:, i] = self._error_metric(nondp_df, dp_trial_df)

        # Aggregate errors
        results = combined_df[query_columns].copy()
        results[NAME_COLUMN] = self._name
        results[ERROR_COLUMN] = self._aggregation(errors)
        return results


class ErrorReport:
    """Creates error reports from multiple trials of an algorithm."""

    def __init__(self, runner: Runner):
        """Constructor.

        Args:
            runner: Runner used to create output files.
        """
        self._runner = runner

    def _create_combined_df(self, eps: float, trials: int) -> pd.DataFrame:
        """Return a combined DataFrame containing all trials.

        Combined DataFrame uses the non_dp algorithm as a base, and statistics
        columns from dp trials are added with with '_0', '_1'... as a suffix.

        Args:
            eps: The epsilon budget for each run.
            trials: The number of trials to run.
        """
        combined_df = self._runner.run_nondp()
        for i in range(trials):
            df = self._runner.run_dp(eps, seed=i)
            combined_df = pd.merge(
                combined_df,
                df,
                how="outer",
                on=self._runner.query_columns,
                suffixes=("", f"_{i}"),
            )
        combined_df = combined_df.reset_index(drop=True)
        return combined_df

    def __call__(
        self, metrics: List[AggregatedErrorMetric], eps: float, trials: int
    ) -> pd.DataFrame:
        """Return error report.

        Args:
            metrics: The aggregated errors to compute.
            eps: The epsilon budget for each run.
            trials: The number of times to run the DP algorithm. Uses seed=0 for first
                run, seed=1 for second run, etc.
        """
        combined_df = self._create_combined_df(eps, trials)
        dfs: List[pd.DataFrame] = []
        for metric in metrics:
            df = metric(
                combined_df,
                self._runner.query_columns,
                self._runner.result_columns,
                trials,
            )
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


def suppress_warnings(func: Callable) -> Callable:
    """Returns a wrapped function with suppressed warnings.

    Useful for functions such as np.nanmean and np.nanmedian which return nan for
    all-NaN slices, which is often the desired behavior, but raise a warning.

    Args:
        func: The function to wrap.
    """

    @wraps(func)
    def silent_func(*args, **kwargs) -> Any:
        """Silent version of func."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return silent_func
