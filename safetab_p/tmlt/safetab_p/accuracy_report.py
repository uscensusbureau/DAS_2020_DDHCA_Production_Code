"""Utilities for comparing the results from SafeTab-P against the ground truth.

As this report uses the ground truth counts, it violates differential privacy,
and should not be run using sensitive data. Rather, its purpose is to test
SafeTab-P on non-sensitive or synthetic datasets to help tune the algorithms and
to predict the performance on private data.
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

import functools
import itertools
import json
import math
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tmlt.common.io_helpers import read_csv, to_csv_with_create_dir
from tmlt.safetab_p.paths import INPUT_CONFIG_DIR
from tmlt.safetab_p.safetab_p_analytics import execute_plan_p_analytics
from tmlt.safetab_p.target_counts_p import create_ground_truth_p
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    get_augmented_df,
    safetab_input_reader,
)

WORKLOAD_SIZE_TO_NAME = {
    1: "Total",
    8: "Sex x Age(4)",
    18: "Sex x Age(9)",
    46: "Sex x Age(23)",
}
"""A dictionary to infer the workload used for a particular population group."""

WORKLOAD_NAME_TO_SIZE = {v: k for k, v in WORKLOAD_SIZE_TO_NAME.items()}
"""Inverse mapping of :data:`~.WORKLOAD_SIZE_TO_NAME`."""


def create_error_report_p(
    noisy_path: str, ground_truth_path: str, parameters_path: str, output_path: str
):
    """Create an error report from a single run of SafeTab-P.

    The error report is saved to {output_path}/error_report.csv

    Args:
        noisy_path: The output directory from a noisy run.
        ground_truth_path: The output directory from a ground truth run.
        parameters_path: the path containing the iteration files (used to gather
            additional information about iteration codes).
        output_path: the path where the output should be saved.
    """
    t1 = get_augmented_df("t1", noisy_path, ground_truth_path)
    t2 = get_augmented_df("t2", noisy_path, ground_truth_path)

    # Drop all additional rows created as part of T1/T2 postprocessing (total in
    # t1 and sex marginal in t2).
    t2_sex_marginal_rows = t2.query("AGESTART == '*' & AGEEND == '*'")
    t2_sex_marginal_rows = t2_sex_marginal_rows[
        ["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]
    ]
    t1 = (
        t1.merge(
            t2_sex_marginal_rows,
            on=["REGION_ID", "REGION_TYPE", "ITERATION_CODE"],
            how="left",
            indicator=True,
        )
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    t2 = t2.query("AGESTART != '*' & AGEEND != '*'")

    # Add missing T2 columns to T1.
    t1["AGESTART"] = "*"
    t1["AGEEND"] = "*"
    t1["SEX"] = "*"

    augmented_t1_t2 = t2.append(t1)

    # Compute the size of each population group.
    error_report = (
        augmented_t1_t2.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"])
        .agg({"GROUND_TRUTH": sum})
        .rename(columns={"GROUND_TRUTH": "Population group size"})
        .reset_index()
    )

    # Compute total absolute error in each workload by aggregating over each workload.
    augmented_t1_t2["Error"] = abs(
        augmented_t1_t2["NOISY"] - augmented_t1_t2["GROUND_TRUTH"]
    )
    error_report = (
        augmented_t1_t2.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"])
        .agg({"Error": sum})
        .rename(columns={"Error": "Total absolute error"})
        .reset_index()
        .merge(error_report)
    )

    # Compute total squared error in each workload
    augmented_t1_t2["Squared error"] = augmented_t1_t2["Error"] ** 2
    error_report = (
        augmented_t1_t2.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"])
        .agg({"Squared error": sum})
        .rename(columns={"Squared error": "Total squared error"})
        .reset_index()
        .merge(error_report)
    )

    # Use number of rows in results to map to workload name
    error_report = (
        augmented_t1_t2.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"])
        .size()
        .map(WORKLOAD_SIZE_TO_NAME)
        .to_frame(name="Workload selected")
        .reset_index()
        .merge(error_report)
    )

    # Load DETAILED_ONLY and COARSE_ONLY info, and join with aggregated_error_report.
    ethnicity_iterations_df = read_csv(
        os.path.join(parameters_path, "ethnicity-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
    )
    race_iterations_df = read_csv(
        os.path.join(parameters_path, "race-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
    )
    iterations_df = pd.concat([ethnicity_iterations_df, race_iterations_df])
    error_report = error_report.merge(iterations_df, how="left")

    # Select and order output columns
    error_report = error_report.loc[
        :,
        [
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
    ]

    to_csv_with_create_dir(
        error_report, os.path.join(output_path, "error_report.csv"), index=False
    )


def run_safetab_p_with_error_report(
    parameters_path: str,
    data_path: str,
    noisy_path: str,
    ground_truth_path: str,
    output_path: str,
    overwrite_config: Optional[Dict] = None,
    us_or_puerto_rico: str = "US",
):
    """Run SafeTab-P and create an error report.

    Args:
        parameters_path: the path containing the config and iterations files.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        noisy_path: the path where the noisy output should be saved.
        ground_truth_path: the path where the ground truth output should be
            saved.
        output_path: the path where the error report should be saved.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
    """
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        reader = config_json[READER_FLAG]
        if us_or_puerto_rico == "US" and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter = config_json[STATE_FILTER_FLAG]
        else:
            state_filter = ["72"]
    input_reader = safetab_input_reader(
        reader=reader,
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-p",
    )
    with tempfile.TemporaryDirectory() as updated_config_dir:
        if validate_input(
            parameters_path=parameters_path,
            input_data_configs_path=INPUT_CONFIG_DIR,
            output_path=updated_config_dir,
            program="safetab-p",
            input_reader=input_reader,
            state_filter=state_filter,
        ):
            execute_plan_p_analytics(
                parameters_path=parameters_path,
                data_path=data_path,
                output_path=noisy_path,
                config_path=updated_config_dir,
                overwrite_config=overwrite_config,
                us_or_puerto_rico=us_or_puerto_rico,
            )
            create_ground_truth_p(
                parameters_path=parameters_path,
                data_path=data_path,
                output_path=ground_truth_path,
                config_path=updated_config_dir,
                safetab_output_paths=[noisy_path],
                us_or_puerto_rico=us_or_puerto_rico,
            )
        create_error_report_p(
            noisy_path=noisy_path,
            ground_truth_path=ground_truth_path,
            parameters_path=parameters_path,
            output_path=output_path,
        )


def run_full_error_report_p(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    trials: int,
    overwrite_config: Optional[Dict] = None,
    us_or_puerto_rico: str = "US",
):
    """Run SafeTab-P for multiple trials and create an error report.

    Run SafeTab-P for <trials> trials. Create a ground truth for these runs and
    noisy answers for each run. Finally, create an aggregated error report from the
    single-run reports. This aggregated error report is saved in
    "<output_path>/multi_run_error_report.csv".

    Args:
        parameters_path: The path containing the input files for SafeTab-P.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The path where the output file "multi_run_error_report.csv" is
            saved.
        config_path: The location of the directory containing the schema files.
        trials: The number of trials to run for each privacy parameter.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
    """
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        reader = config_json[READER_FLAG]
        if us_or_puerto_rico == "US" and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter = config_json[STATE_FILTER_FLAG]
        else:
            state_filter = ["72"]
    input_reader = safetab_input_reader(
        reader=reader,
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-p",
    )
    validate_input(
        parameters_path=parameters_path,
        input_data_configs_path=INPUT_CONFIG_DIR,
        output_path=config_path,
        program="safetab-p",
        input_reader=input_reader,
        state_filter=state_filter,
    )

    # All paths corresponding to safetab runs.
    single_run_paths: List[str] = []

    # Run safetab for <trials> trials and save the results.
    for trial in range(trials):
        dir_name = f"trial_{trial}"
        single_run_path = os.path.join(output_path, "single_runs", dir_name)
        single_run_paths.append(single_run_path)
        execute_plan_p_analytics(
            parameters_path=parameters_path,
            data_path=data_path,
            output_path=single_run_path,
            config_path=config_path,
            overwrite_config=overwrite_config,
            us_or_puerto_rico=us_or_puerto_rico,
        )

    # Create ground truth using all safetab runs for all values of privacy parameter.
    ground_truth_path = os.path.join(output_path, "ground_truth")
    create_ground_truth_p(
        parameters_path=parameters_path,
        data_path=data_path,
        output_path=ground_truth_path,
        config_path=config_path,
        safetab_output_paths=single_run_paths,
        us_or_puerto_rico=us_or_puerto_rico,
    )

    # Create the aggregated error report.
    multi_run_path = os.path.join(output_path, "full_error_report")
    create_aggregated_error_report_p(
        single_run_paths=single_run_paths,
        parameters_path=parameters_path,
        ground_truth_path=ground_truth_path,
        output_path=multi_run_path,
    )


def create_aggregated_error_report_p(
    single_run_paths: List[str],
    parameters_path: str,
    ground_truth_path: str,
    output_path: str,
):
    """Create an error report with population groups aggregated by size.

    Create an error report with population groups aggregated by size and grouped by
    DETAILED_ONLY, iteration level, and geography level.

    Args:
        single_run_paths: A list of paths to directories containing single runs of
            SafeTab-p.
        parameters_path: The path containing the input files.
        ground_truth_path: The path containing the ground truth t1 and t2 counts.
        output_path: The path where the output file "multi_run_error_report.csv" is
            saved.
    """
    dfs = []
    for trial, single_run_path in enumerate(single_run_paths):
        t1 = get_augmented_df("t1", single_run_path, ground_truth_path)
        t2 = get_augmented_df("t2", single_run_path, ground_truth_path)

        # Drop all additional rows created as part of T1/T2 postprocessing (total
        # in t1 and sex marginal in t2).
        t2_sex_marginal_rows = t2.query("AGESTART == '*' & AGEEND == '*'")
        t2_sex_marginal_rows = t2_sex_marginal_rows[
            ["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]
        ]
        t1 = (
            t1.merge(
                t2_sex_marginal_rows,
                on=["REGION_ID", "REGION_TYPE", "ITERATION_CODE"],
                how="left",
                indicator=True,
            )
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )
        t2 = t2.query("AGESTART != '*' & AGEEND != '*'")

        t2 = t2.drop(columns=["AGESTART", "AGEEND", "SEX"])
        df = t2.append(t1)
        df["Trial"] = trial
        dfs.append(df)

    error_report = pd.concat(dfs)

    # Compute the size of each population group.
    error_report = (
        error_report.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "Trial"])
        .agg({"GROUND_TRUTH": sum})
        .rename(columns={"GROUND_TRUTH": "Population group size"})
        .reset_index()
        .merge(error_report)
    )

    # Compute total absolute error in each workload by aggregating over each workload.
    error_report["Error"] = abs(error_report["NOISY"] - error_report["GROUND_TRUTH"])

    # Use number of rows in results to map to workload name.
    error_report = (
        error_report.groupby(["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "Trial"])
        .size()
        .map(WORKLOAD_SIZE_TO_NAME)
        .to_frame(name="Workload selected")
        .reset_index()
        .merge(error_report)
    )

    # Load DETAILED_ONLY, COARSE_ONLY and LEVEL info, and join with
    # aggregated_error_report.
    ethnicity_iterations_df = read_csv(
        os.path.join(parameters_path, "ethnicity-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY", "LEVEL"],
    )
    race_iterations_df = read_csv(
        os.path.join(parameters_path, "race-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY", "LEVEL"],
    )
    iterations_df = pd.concat([ethnicity_iterations_df, race_iterations_df])
    error_report = error_report.merge(iterations_df, how="left")

    error_report = error_report.rename(columns={"LEVEL": "ITERATION_LEVEL"})

    # Bin population group sizes by powers of 10.
    max_pop_group_size = error_report["Population group size"].max()
    bins = [0] + [
        10 ** i for i in range(0, math.ceil(np.log10(max_pop_group_size)) + 1)
    ]
    error_report["Population group size"] = pd.cut(
        error_report["Population group size"], bins, include_lowest=True
    )

    # Group and calculate the the margin of error (95%).
    grouped_error_report = error_report.groupby(
        [
            "REGION_TYPE",
            "ITERATION_LEVEL",
            "DETAILED_ONLY",
            "Population group size",
            "Workload selected",
        ]
    )
    compute_95_moe = functools.partial(np.quantile, q=0.95, interpolation="linear")
    aggregated_error_report = (
        grouped_error_report.agg({"Error": compute_95_moe})
        .rename(columns={"Error": "MOE"})
        .reset_index()
    )

    # Compute the number of times the workload was selected.
    aggregated_error_report = (
        grouped_error_report.size()
        .reset_index(name="Times workload selected")
        .merge(aggregated_error_report)
    )
    aggregated_error_report["Times workload selected"] /= aggregated_error_report[
        "Workload selected"
    ].map(WORKLOAD_NAME_TO_SIZE)
    # Sanity check that workload selected counts are not fractional, and convert to int.
    times_workload_selected_as_int = aggregated_error_report[
        "Times workload selected"
    ].astype(int)
    assert (
        aggregated_error_report["Times workload selected"]
        == times_workload_selected_as_int
    ).all()
    aggregated_error_report["Times workload selected"] = times_workload_selected_as_int

    # Pivot table to have error/count columns for each workload. pivot_table is used
    # because pivot does not accept list for the index. aggfunc is max, but each group
    # only contains one element so there will be no aggregation.
    aggregated_error_report = aggregated_error_report.pivot_table(
        index=[
            "REGION_TYPE",
            "ITERATION_LEVEL",
            "DETAILED_ONLY",
            "Population group size",
        ],
        columns="Workload selected",
        values=["Times workload selected", "MOE"],
        aggfunc="max",
    )
    aggregated_error_report.columns = aggregated_error_report.columns.to_flat_index()
    aggregated_error_report = aggregated_error_report.rename(
        columns=lambda column_name: (
            "MOE of " + column_name[1]
            if column_name[0] == "MOE"
            else "Average proportion of pop groups with workload = " + column_name[1]
        )
    )
    aggregated_error_report = aggregated_error_report.reset_index()

    # Add any missing columns.
    error_column_names = ["MOE of " + name for name in WORKLOAD_SIZE_TO_NAME.values()]
    frequency_column_names = [
        "Average proportion of pop groups with workload = " + name
        for name in WORKLOAD_SIZE_TO_NAME.values()
    ]
    for frequency_column_name in frequency_column_names:
        if frequency_column_name not in aggregated_error_report.columns:
            aggregated_error_report[frequency_column_name] = 0
    for error_column_name in error_column_names:
        if error_column_name not in aggregated_error_report.columns:
            aggregated_error_report[error_column_name] = np.NaN

    # Compute the (incorrect) number of population groups falling in each bucket. This
    # needs to be divided by the number of trials for each privacy parameter.
    aggregated_error_report[
        "Number of population groups"
    ] = aggregated_error_report.loc[:, frequency_column_names].sum(axis="columns")

    # Compute the frequency for each workload
    aggregated_error_report.loc[
        :, frequency_column_names
    ] = aggregated_error_report.loc[:, frequency_column_names].div(
        aggregated_error_report["Number of population groups"], axis=0
    )
    # Remove rows without population groups.
    aggregated_error_report = aggregated_error_report.loc[
        aggregated_error_report["Number of population groups"] != 0
    ]

    # Remove the number of population groups, as it's not really a meaningful number
    # with suppression.
    aggregated_error_report.drop("Number of population groups", axis=1)

    # Sort columns
    aggregated_error_report = aggregated_error_report.loc[
        :,
        ["REGION_TYPE", "ITERATION_LEVEL", "DETAILED_ONLY", "Population group size"]
        + list(
            itertools.chain.from_iterable(
                zip(frequency_column_names, error_column_names)
            )
        ),
    ]

    # Convert imprecise intervals to precise intervals as strings.
    aggregated_error_report["Population group size"] = aggregated_error_report[
        "Population group size"
    ].astype(str)
    aggregated_error_report.loc[
        aggregated_error_report["Population group size"] == "(-0.001, 1.0]",
        "Population group size",
    ] = "[0.0, 1.0]"

    # Format floats.
    for column in aggregated_error_report.columns[4:]:
        aggregated_error_report[column] = aggregated_error_report[column].map(
            lambda x: "{0:.2E}".format(x) if 0 < x < 0.01 else "{0:.2f}".format(x)
        )

    to_csv_with_create_dir(
        aggregated_error_report,
        os.path.join(output_path, "multi_run_error_report.csv"),
        index=False,
    )
