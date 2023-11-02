"""Miscellaneous tools for SafeTab."""

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
import glob
import os
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from pyspark.sql.types import Row
from typing_extensions import Literal

from tmlt.common.configuration import Config
from tmlt.common.io_helpers import multi_read_csv
from tmlt.common.validation import validate_file
from tmlt.safetab_utils.csv_reader import CSVHReader, CSVPReader

try:
    from phsafe_safetab_reader.safetab_cef_reader import CEFHReader, CEFPReader
except ImportError as e:
    try:
        from tmlt.mock_cef_reader.safetab_cef_reader import (  # pylint: disable=ungrouped-imports
            CEFHReader,
            CEFPReader,
        )
    except ImportError:
        raise ImportError(
            "Failed to import CEFPReader & CEFHReader from module "
            f"safetab_cef_reader: {e} Please verify the safetab_cef_reader module "
            "has been placed in your python path."
        )

READER_FLAG = "reader"
"""Key in the config.json that indicates the reader being used."""

STATE_FILTER_FLAG = "state_filter_us"
"""Key in the config.json that indicates states to be kept in us runs."""


def file_empty(filename: str) -> bool:
    """Returns whether the file is empty or not.

    Args:
        filename: file to test.
    """
    return os.stat(filename).st_size == 0


def safetab_input_reader(
    reader: str,
    data_path: str,
    state_filter: List[str],
    program: Literal["safetab-h", "safetab-p"],
) -> Union[CSVHReader, CSVPReader, CEFHReader, CEFPReader]:
    """Returns an AbstractSafeTab*Reader configured to read the program's input.

    Chooses the appropriate subclass based on the program type (safetab-h vs
    safetab-p) and the configuration (cef reader vs csv).

    Args:
        reader: Reader to use. "csv" or "cef"
        data_path: For CEF readers: the path to the reader config. For
            CSV readers: the path to the directory containing the geography and
            person/unit files.
        state_filter: States to include.
        program: The program name. "safetab-h" or "safetab-p"
    """
    if reader not in ["csv", "cef"]:
        raise ValueError("Invalid config: 'reader' must be one of: csv cef")
    if reader == "csv" and program == "safetab-h":
        return CSVHReader(data_path, state_filter)
    if reader == "csv" and program == "safetab-p":
        return CSVPReader(data_path, state_filter)
    if reader == "cef" and program == "safetab-h":
        return CEFHReader(data_path, state_filter)
    if reader == "cef" and program == "safetab-p":
        return CEFPReader(data_path, state_filter)
    raise ValueError(
        "Unexpected input reader configuration, not a combination of {csv, cef} x"
        " {safetab-h, safetab-p}"
    )


def create_augmenting_map(
    f: Callable[[Row], List[Dict[str, str]]]
) -> Callable[[Row], List[Row]]:
    """Create an augmenting map for a non-augmenting map.

    Args:
        f: A function mapping a row to list of rows that should be converted to augment
            the original row.
    """

    def augmenting_map(row: Row) -> List[Row]:
        mapped_rows = f(row)
        augmented_rows: List[Row] = list()
        for r in mapped_rows:
            augmented_row_dict = {**row.asDict(), **r}
            augmented_rows.append(Row(**augmented_row_dict))
        return augmented_rows

    return augmenting_map


def get_augmented_df(
    name: str, noisy_path: str, ground_truth_path: str
) -> pd.DataFrame:
    """Return a dataframe containing both noisy and ground_truth counts.

    Args:
        name: Name of the file. Either "t1", "t2", "t3", or "t4".
        noisy_path: The output directory from a noisy run of SafeTab-P or
            SafeTab-H.
        ground_truth_path: The output directory from a ground-truth run of
            SafeTab-P or SafeTab-H.

    Returns:
        A dataframe containing all of the expected columns from `Appendix A`,
        minus "COUNT". The dataframe is augmented with the following columns:

        * NOISY: The counts from the mechanism, as integers.
        * GROUND_TRUTH: The ground truth counts, as integers.
    """
    noisy_df = multi_read_csv(os.path.join(noisy_path, name), sep="|", dtype=str)
    ground_truth_df = multi_read_csv(
        os.path.join(ground_truth_path, name), sep="|", dtype=str
    )
    noisy_df = noisy_df.rename(columns={"COUNT": "NOISY"})
    ground_truth_df = ground_truth_df.rename(columns={"COUNT": "GROUND_TRUTH"})
    augmented_df = pd.merge(noisy_df, ground_truth_df, how="left")
    augmented_df["NOISY"] = augmented_df["NOISY"].astype(int)
    augmented_df["GROUND_TRUTH"] = augmented_df["GROUND_TRUTH"].astype(int)
    return augmented_df


def validate_directory_single_config(
    input_dir: str, config_file: str, **kwargs: Any
) -> bool:
    """Validate all files in a directory against a single config using validate_file.

    The passed directory should contain only files that should be validated.

    Args:
        input_dir: The directory containing the csv files to validate.
        config_file: The filename of the config to validate against.
        kwargs: Keyword arguments to pass to validate_file.
    """
    filenames = glob.glob(f"{input_dir}/*.csv")
    config = Config.load_json(config_file)
    okay = True
    for filename in filenames:
        # SPARK-26208: spark DF write.csv creates empty part file without header
        # which fails on output validation.
        if not file_empty(filename):
            okay &= validate_file(filename, config, **kwargs)
    return okay
